"""Re-split YOLO dataset into stratified 80/10/10 train/val/test.

The original Roboflow export has too few val/test samples (16 val, 8 test)
with some classes having 0-2 validation samples. This script merges all
images into one pool and re-splits with class stratification.

Usage:
    python scripts/resplit_dataset.py

    # Custom split ratios:
    python scripts/resplit_dataset.py --train 0.85 --val 0.10 --test 0.05

    # Dry run (show stats without moving files):
    python scripts/resplit_dataset.py --dry-run

The original dataset is backed up to data/datasets/league-yolo-original/
before re-splitting.
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "league-yolo"
BACKUP_DIR = PROJECT_ROOT / "data" / "datasets" / "league-yolo-original"

SPLITS = ["train", "valid", "test"]


def parse_args():
    p = argparse.ArgumentParser(description="Re-split YOLO dataset with stratification")
    p.add_argument("--train", type=float, default=0.80, help="Train fraction (default: 0.80)")
    p.add_argument("--val", type=float, default=0.10, help="Val fraction (default: 0.10)")
    p.add_argument("--test", type=float, default=0.10, help="Test fraction (default: 0.10)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--dry-run", action="store_true", help="Show stats without moving files")
    return p.parse_args()


def get_classes_for_image(label_path: Path) -> set[int]:
    """Read YOLO label file and return set of class IDs present."""
    classes = set()
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                classes.add(int(parts[0]))
    return classes


def collect_all_images(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """Collect all (image_path, label_path) pairs from all splits."""
    pairs = []
    for split in SPLITS:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                pairs.append((img_path, lbl_path))
    return pairs


def stratified_split(
    pairs: list[tuple[Path, Path]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    """Split image/label pairs with class stratification.

    Uses the dominant (most common) class in each image as the stratification
    key. Images with no labels go into a separate "no_label" stratum.
    """
    # Group by dominant class
    strata: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for img_path, lbl_path in pairs:
        classes = get_classes_for_image(lbl_path)
        if classes:
            # Use the most frequent class as stratum key.
            # For ties, use the lowest class ID for consistency.
            counter = Counter()
            if lbl_path.exists():
                for line in lbl_path.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if parts:
                        counter[int(parts[0])] += 1
            dominant = counter.most_common(1)[0][0] if counter else 0
            strata[str(dominant)].append((img_path, lbl_path))
        else:
            strata["no_label"].append((img_path, lbl_path))

    rng = random.Random(seed)
    result: dict[str, list[tuple[Path, Path]]] = {"train": [], "valid": [], "test": []}

    for stratum_key in sorted(strata.keys()):
        items = strata[stratum_key]
        rng.shuffle(items)

        n = len(items)
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        n_test = max(1, round(n * (1 - train_frac - val_frac))) if n >= 3 else 0
        # Ensure we don't exceed total
        if n_val + n_test >= n:
            n_val = min(1, n - 1) if n > 1 else 0
            n_test = min(1, n - n_val - 1) if n > 2 else 0

        result["test"].extend(items[:n_test])
        result["valid"].extend(items[n_test:n_test + n_val])
        result["train"].extend(items[n_test + n_val:])

    return result


def print_stats(
    splits: dict[str, list[tuple[Path, Path]]],
    class_names: list[str] | None = None,
):
    """Print per-split and per-class statistics."""
    print(f"\n{'Split':<8} {'Images':>8}")
    print("-" * 20)
    for split_name in ["train", "valid", "test"]:
        print(f"{split_name:<8} {len(splits[split_name]):>8}")
    print(f"{'TOTAL':<8} {sum(len(v) for v in splits.values()):>8}")

    # Per-class counts per split
    print(f"\n{'Class':<30} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}")
    print("-" * 55)

    all_classes: set[int] = set()
    class_counts: dict[str, Counter] = {s: Counter() for s in ["train", "valid", "test"]}

    for split_name in ["train", "valid", "test"]:
        for _, lbl_path in splits[split_name]:
            classes = get_classes_for_image(lbl_path)
            for cls_id in classes:
                class_counts[split_name][cls_id] += 1
                all_classes.add(cls_id)

    for cls_id in sorted(all_classes):
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"class_{cls_id}"
        tr = class_counts["train"][cls_id]
        va = class_counts["valid"][cls_id]
        te = class_counts["test"][cls_id]
        print(f"{name:<30} {tr:>6} {va:>6} {te:>6} {tr+va+te:>6}")


def main():
    args = parse_args()

    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset not found at {DATASET_DIR}")
        raise SystemExit(1)

    # Load class names from data.yaml if available
    class_names = None
    data_yaml = DATASET_DIR / "data.yaml"
    if data_yaml.exists():
        import yaml
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        class_names = cfg.get("names", None)
        if isinstance(class_names, dict):
            max_id = max(class_names.keys())
            class_names = [class_names.get(i, f"class_{i}") for i in range(max_id + 1)]

    # Collect all images (try backup dir first if main is empty)
    pairs = collect_all_images(DATASET_DIR)
    if len(pairs) == 0 and BACKUP_DIR.exists():
        print("Main dataset empty, collecting from backup...")
        pairs = collect_all_images(BACKUP_DIR)
    print(f"Found {len(pairs)} total images across all splits")

    if len(pairs) == 0:
        print("ERROR: No images found")
        raise SystemExit(1)

    # Perform stratified split
    splits = stratified_split(pairs, args.train, args.val, args.seed)

    # Print statistics
    print_stats(splits, class_names)

    if args.dry_run:
        print("\n[DRY RUN] No files were moved.")
        return

    # Backup original
    if not BACKUP_DIR.exists():
        print(f"\nBacking up original to {BACKUP_DIR}...")
        shutil.copytree(str(DATASET_DIR), str(BACKUP_DIR))
    else:
        print(f"\nBackup already exists at {BACKUP_DIR}")

    # Remap source paths to the backup directory so they survive deletion
    def remap_to_backup(p: Path) -> Path:
        try:
            rel = p.relative_to(DATASET_DIR)
        except ValueError:
            return p
        return BACKUP_DIR / rel

    for split_name in splits:
        splits[split_name] = [
            (remap_to_backup(img), remap_to_backup(lbl))
            for img, lbl in splits[split_name]
        ]

    # Clear existing splits and re-create
    for split in SPLITS:
        for subdir in ["images", "labels"]:
            d = DATASET_DIR / split / subdir
            if d.exists():
                shutil.rmtree(str(d))
            d.mkdir(parents=True, exist_ok=True)

    # Copy files into new splits
    for split_name in ["train", "valid", "test"]:
        img_dir = DATASET_DIR / split_name / "images"
        lbl_dir = DATASET_DIR / split_name / "labels"
        for img_src, lbl_src in splits[split_name]:
            shutil.copy2(str(img_src), str(img_dir / img_src.name))
            if lbl_src.exists():
                shutil.copy2(str(lbl_src), str(lbl_dir / lbl_src.name))

    print(f"\nDone! Re-split {len(pairs)} images into {DATASET_DIR}/")
    print("Original backed up to:", BACKUP_DIR)


if __name__ == "__main__":
    main()
