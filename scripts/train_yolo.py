"""Train YOLOv8 on League of Legends gameplay dataset and export to ONNX.

GPU setup (RTX 20-series+):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install ultralytics roboflow

Usage:
    # Train YOLOv8s on GPU (recommended defaults):
    python scripts/train_yolo.py

    # Train YOLOv8m for higher accuracy:
    python scripts/train_yolo.py --model yolov8m --epochs 200

    # CPU fallback:
    python scripts/train_yolo.py --device cpu

    # Custom batch size for limited VRAM:
    python scripts/train_yolo.py --batch 8

Dataset structure (download from Roboflow in YOLOv8 format):
    data/datasets/league-yolo/
      data.yaml
      train/images/  train/labels/
      valid/images/  valid/labels/
      test/images/   test/labels/
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "league-yolo"
DATA_YAML = DATASET_DIR / "data.yaml"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "models"

# 16 classes from vasyz's dataset (for per-class reporting)
CLASS_NAMES = [
    "enemy_cannon_minion", "enemy_champion", "enemy_inhibitor",
    "enemy_melee_minion", "enemy_nexus", "enemy_ranged_minion",
    "enemy_tower", "freindly_cannon_minion", "freindly_champion",
    "freindly_melee_minion", "freindly_ranged_minion",
    "freindly_super_minion", "friendly_inhibitor", "friendly_nexus",
    "friendly_tower", "played_champion",
]


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on League dataset")
    p.add_argument(
        "--model", default="yolov8s.pt",
        help="Base model: yolov8n.pt, yolov8s.pt, yolov8m.pt (default: yolov8s.pt)",
    )
    p.add_argument("--epochs", type=int, default=150, help="Training epochs (default: 150)")
    p.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    p.add_argument(
        "--device", default="0",
        help="Device: 0 for GPU, cpu for CPU (default: 0)",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience (default: 50)")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    if not DATA_YAML.exists():
        print(f"ERROR: Dataset not found at {DATA_YAML}")
        print("Download from Roboflow in YOLOv8 format and extract to:")
        print(f"  {DATASET_DIR}/")
        raise SystemExit(1)

    from ultralytics import YOLO

    # Determine model variant name for ONNX output (e.g., yolov8s -> yolov8s_lol.onnx)
    model_variant = Path(args.model).stem  # "yolov8s" from "yolov8s.pt"
    onnx_output = MODEL_OUTPUT_DIR / f"{model_variant}_lol.onnx"

    # Load base model
    model = YOLO(args.model)

    print(f"Model:   {args.model}")
    print(f"Device:  {args.device}")
    print(f"Epochs:  {args.epochs} (patience={args.patience})")
    print(f"Batch:   {args.batch}")
    print(f"ImgSz:   {args.imgsz}")
    print(f"Output:  {onnx_output}")
    print()

    # Train with augmentation tuned for League gameplay
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=f"{model_variant}-league",
        exist_ok=True,
        resume=args.resume,
        # Augmentation — tuned for League:
        # degrees=0: HUD elements are always upright, no rotation
        # scale=0.5: handle zoom variation in different replay clients
        # mixup=0.15: regularization for small dataset
        # copy_paste=0.1: synthetic diversity for minion/champion instances
        degrees=0.0,
        scale=0.5,
        mixup=0.15,
        copy_paste=0.1,
    )

    # Load best weights and evaluate
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best_weights}")

    val_model = YOLO(str(best_weights))
    metrics = val_model.val(data=str(DATA_YAML))

    print(f"\n{'='*60}")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"{'='*60}")

    # Per-class AP50
    print(f"\n{'Class':<30} {'AP50':>8} {'AP50-95':>8} {'Instances':>10}")
    print("-" * 60)
    ap50_per_class = metrics.box.ap50
    ap_per_class = metrics.box.ap
    for i, name in enumerate(CLASS_NAMES):
        if i < len(ap50_per_class):
            print(f"{name:<30} {ap50_per_class[i]:>8.3f} {ap_per_class[i]:>8.3f}")
    print()

    # Export to ONNX
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    val_model.export(format="onnx", imgsz=args.imgsz)

    exported_onnx = best_weights.with_suffix(".onnx")
    if exported_onnx.exists():
        import shutil
        shutil.copy2(str(exported_onnx), str(onnx_output))
        print(f"ONNX model saved to: {onnx_output}")
        print(f"Size: {onnx_output.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"WARNING: Expected ONNX at {exported_onnx} not found.")
        print("Check the runs/ directory for the exported model.")


if __name__ == "__main__":
    main()
