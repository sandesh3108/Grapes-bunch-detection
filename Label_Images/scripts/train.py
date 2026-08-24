"""
YOLO Training Script for Label_Images Dataset
"""

import sys
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model on Label_Images dataset")
    parser.add_argument("--data", type=str, default="data/processed/v001/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model (e.g. yolo11n.pt, yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: 'ultralytics' package is not installed. Run 'python -m pip install ultralytics' first.")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project_root / data_path

    if not data_path.exists():
        print(f"Error: dataset configuration file not found at: {data_path}")
        sys.exit(1)

    print(f"[Train] Initializing YOLO model: {args.model}")
    print(f"[Train] Dataset configuration: {data_path}")
    print(f"[Train] Epochs: {args.epochs}, Batch size: {args.batch}, Image size: {args.imgsz}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(project_root / "runs" / "detect"),
        name="train",
        exist_ok=True
    )

    best_weights = project_root / "runs" / "detect" / "train" / "weights" / "best.pt"
    print(f"\n[Train] Training complete!")
    print(f"[Train] Best model saved at: {best_weights}")

if __name__ == "__main__":
    main()
