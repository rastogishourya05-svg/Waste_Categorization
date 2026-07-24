"""
Fine-tune YOLOv8 on a custom waste/trash dataset.

Usage:
    python src/training/train.py --data ../../data/annotations/data.yaml --epochs 100

Run this on Colab/Kaggle if you don't have a local GPU - just upload the
data/ folder (or mount Drive) and point --data at the yaml there.
"""

import argparse
from ultralytics import YOLO


def train(data_yaml: str, epochs: int, imgsz: int, batch: int, base_model: str):
    # Load a pretrained checkpoint and fine-tune - do NOT train from scratch,
    # you don't have the data volume for that.
    model = YOLO(base_model)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,          # early stopping if val mAP plateaus
        project="runs/waste-detect",
        name="train",
        exist_ok=True,
        augment=True,          # built-in mosaic/flip/hsv augmentation
        val=True,
    )

    print("Training complete.")
    print(f"Best weights saved to: runs/waste-detect/train/weights/best.pt")
    return results


def validate(weights: str, data_yaml: str):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../../data/annotations/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base-model", type=str, default="yolov8s.pt",
                         help="yolov8n.pt (fastest) | yolov8s.pt (balanced) | yolov8m.pt (more accurate, slower)")
    args = parser.parse_args()

    train(args.data, args.epochs, args.imgsz, args.batch, args.base_model)
