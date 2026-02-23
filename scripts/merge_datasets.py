"""Merge additional Roboflow datasets into the main training dataset.

Downloads datasets from Roboflow, remaps class indices to match our 16-class
schema, prefixes filenames to avoid collisions, and updates data.yaml.

Usage:
    # Merge a dataset by workspace/project/version:
    python scripts/merge_datasets.py vasyz/leagueoflegends-kvjwx/4

    # Merge multiple:
    python scripts/merge_datasets.py workspace1/project1/1 workspace2/project2/2

    # Dry run:
    python scripts/merge_datasets.py --dry-run vasyz/leagueoflegends-kvjwx/4

Requires ROBOFLOW_API_KEY in .env or environment.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "league-yolo"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "datasets" / "downloads"

# Our canonical class list (must match detector.py CLASS_NAMES exactly)
OUR_CLASSES = [
    "enemy_cannon_minion",     # 0
    "enemy_champion",          # 1
    "enemy_inhibitor",         # 2
    "enemy_melee_minion",      # 3
    "enemy_nexus",             # 4
    "enemy_ranged_minion",     # 5
    "enemy_tower",             # 6
    "freindly_cannon_minion",  # 7  (sic — matches dataset spelling)
    "freindly_champion",       # 8  (sic)
    "freindly_melee_minion",   # 9  (sic)
    "freindly_ranged_minion",  # 10 (sic)
    "freindly_super_minion",   # 11 (sic)
    "friendly_inhibitor",      # 12
    "friendly_nexus",          # 13
    "friendly_tower",          # 14
    "played_champion",         # 15
]

OUR_CLASS_TO_IDX = {name: i for i, name in enumerate(OUR_CLASSES)}

# Common class name variations that map to our classes
ALIAS_MAP = {
    "friendly_cannon_minion": "freindly_cannon_minion",
    "friendly_champion": "freindly_champion",
    "friendly_melee_minion": "freindly_melee_minion",
    "friendly_ranged_minion": "freindly_ranged_minion",
    "friendly_super_minion": "freindly_super_minion",
}


def parse_args():
    p = argparse.ArgumentParser(description="Merge Roboflow datasets")
    p.add_argument("datasets", nargs="+", help="workspace/project/version identifiers")
    p.add_argument("--dry-run", action="store_true", help="Show what would be merged")
    p.add_argument("--format", default="yolov8", help="Download format (default: yolov8)")
    return p.parse_args()


def load_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ROBOFLOW_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    print("ERROR: ROBOFLOW_API_KEY not found")
    sys.exit(1)


def download_dataset(api_key: str, dataset_spec: str, fmt: str) -> Path:
    """Download a dataset from Roboflow. Returns the extracted directory."""
    from roboflow import Roboflow

    parts = dataset_spec.split("/")
    if len(parts) != 3:
        print(f"ERROR: Expected workspace/project/version, got: {dataset_spec}")
        sys.exit(1)

    workspace, project, version = parts[0], parts[1], int(parts[2])

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ds = proj.version(version)

    dest = DOWNLOAD_DIR / f"{workspace}_{project}_v{version}"
    dest.mkdir(parents=True, exist_ok=True)
    ds.download(fmt, location=str(dest))

    return dest


def load_class_list(dataset_path: Path) -> list[str]:
    """Load class names from data.yaml."""
    import yaml
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        # Check subdirectories
        for sub in dataset_path.iterdir():
            if sub.is_dir() and (sub / "data.yaml").exists():
                data_yaml = sub / "data.yaml"
                break

    if not data_yaml.exists():
        print(f"ERROR: No data.yaml found in {dataset_path}")
        return []

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names", [])
    if isinstance(names, dict):
        max_id = max(names.keys())
        return [names.get(i, f"class_{i}") for i in range(max_id + 1)]
    return list(names)


def build_remap(source_classes: list[str]) -> dict[int, int]:
    """Build mapping from source class indices to our class indices.

    Returns dict mapping source_idx -> our_idx. Unmapped classes are excluded.
    """
    remap = {}
    for src_idx, src_name in enumerate(source_classes):
        # Direct match
        if src_name in OUR_CLASS_TO_IDX:
            remap[src_idx] = OUR_CLASS_TO_IDX[src_name]
            continue
        # Check aliases
        canonical = ALIAS_MAP.get(src_name)
        if canonical and canonical in OUR_CLASS_TO_IDX:
            remap[src_idx] = OUR_CLASS_TO_IDX[canonical]
            continue
        # Case-insensitive match
        lower = src_name.lower().strip()
        for our_name, our_idx in OUR_CLASS_TO_IDX.items():
            if lower == our_name.lower():
                remap[src_idx] = our_idx
                break

    return remap


def merge_split(
    src_path: Path,
    split: str,
    prefix: str,
    remap: dict[int, int],
    dry_run: bool,
) -> tuple[int, int]:
    """Merge one split's images and labels. Returns (images_added, images_skipped)."""
    src_img_dir = src_path / split / "images"
    src_lbl_dir = src_path / split / "labels"
    dst_img_dir = DATASET_DIR / "train" / "images"  # merge everything into train
    dst_lbl_dir = DATASET_DIR / "train" / "labels"

    if not src_img_dir.exists():
        return 0, 0

    added = 0
    skipped = 0

    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue

        lbl_path = src_lbl_dir / (img_path.stem + ".txt")

        # Remap labels
        new_lines = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                src_cls = int(parts[0])
                if src_cls not in remap:
                    continue  # skip unmapped classes
                parts[0] = str(remap[src_cls])
                new_lines.append(" ".join(parts))

        if not new_lines:
            skipped += 1
            continue

        # Prefixed filename to avoid collisions
        dst_name = f"{prefix}_{img_path.name}"
        dst_lbl_name = f"{prefix}_{img_path.stem}.txt"

        if not dry_run:
            shutil.copy2(str(img_path), str(dst_img_dir / dst_name))
            (dst_lbl_dir / dst_lbl_name).write_text("\n".join(new_lines) + "\n")

        added += 1

    return added, skipped


def main():
    args = parse_args()
    api_key = load_api_key()

    total_added = 0
    total_skipped = 0

    for dataset_spec in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_spec}")
        print("=" * 60)

        # Download
        if not args.dry_run:
            ds_path = download_dataset(api_key, dataset_spec, args.format)
        else:
            # Check if already downloaded
            parts = dataset_spec.split("/")
            ds_path = DOWNLOAD_DIR / f"{parts[0]}_{parts[1]}_v{parts[2]}"
            if not ds_path.exists():
                print(f"  [DRY RUN] Would download {dataset_spec}")
                continue

        # Load source class list
        source_classes = load_class_list(ds_path)
        if not source_classes:
            print("  Skipping: no class list found")
            continue

        print(f"  Source classes ({len(source_classes)}):")
        for i, name in enumerate(source_classes):
            print(f"    {i}: {name}")

        # Build class remap
        remap = build_remap(source_classes)
        print(f"\n  Mapped {len(remap)}/{len(source_classes)} classes:")
        for src_idx, dst_idx in sorted(remap.items()):
            print(f"    {source_classes[src_idx]} ({src_idx}) -> {OUR_CLASSES[dst_idx]} ({dst_idx})")

        unmapped = set(range(len(source_classes))) - set(remap.keys())
        if unmapped:
            print(f"  Unmapped: {', '.join(source_classes[i] for i in sorted(unmapped))}")

        # Merge all splits into train (will be re-split later with resplit_dataset.py)
        prefix = dataset_spec.replace("/", "_")
        for split in ["train", "valid", "test"]:
            added, skipped = merge_split(ds_path, split, prefix, remap, args.dry_run)
            if added > 0 or skipped > 0:
                print(f"  {split}: +{added} images ({skipped} skipped)")
            total_added += added
            total_skipped += skipped

    print(f"\n{'='*60}")
    print(f"Total: +{total_added} images merged, {total_skipped} skipped")
    if not args.dry_run:
        print("\nRun scripts/resplit_dataset.py to re-split into train/val/test")
    else:
        print("\n[DRY RUN] No files were modified.")


if __name__ == "__main__":
    main()
