"""
Training Script for YOLO Layout Detection
Trains an Ultralytics YOLO model on document layout bounding boxes.
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_yolo(
    data_yaml: str = "config/layout_data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 8,
    img_size: int = 1024,
    device: str = "0",
    project_name: str = "runs/train",
    experiment_name: str = "bangla_layout",
    patience: int = 15,
    save_weights_dir: str = "weights"
):
    """
    Trains YOLO model with parameters optimized for document layout detection.
    High resolution (img_size=1024) is recommended to capture thin text lines.
    """
    data_path = Path(data_yaml).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_path}")
        
    print(f"Initializing YOLO model: {model_name}")
    model = YOLO(model_name)
    
    print(f"Starting training on {data_path} for {epochs} epochs...")
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project_name,
        name=experiment_name,
        patience=patience,
        save=True,
        plots=True,
        # Document layout augmentations (avoid aggressive shearing/cutting)
        degrees=1.0,       # subtle rotation
        shear=0.0,         # disable severe shear
        perspective=0.0,   # documents are mostly planar
        flipud=0.0,        # no upside down text
        fliplr=0.0         # no mirrored text
    )
    
    # Save best weights to dedicated weights/ directory
    weights_dir = Path(save_weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate best.pt using results.save_dir or fallback path
    best_pt = None
    if hasattr(results, "save_dir") and results.save_dir:
        candidate = Path(results.save_dir) / "weights" / "best.pt"
        if candidate.exists():
            best_pt = candidate
    if not best_pt or not best_pt.exists():
        candidate = Path(project_name) / experiment_name / "weights" / "best.pt"
        if candidate.exists():
            best_pt = candidate
            
    if best_pt and best_pt.exists():
        target_pt = weights_dir / "best_layout.pt"
        import shutil
        shutil.copyfile(best_pt, target_pt)
        print(f"\n[SUCCESS] Best model weights successfully copied to: {target_pt}")
    else:
        print(f"\n[NOTE] Training completed. Check {project_name}/{experiment_name}/weights for checkpoints.")
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model for layout detection.")
    parser.add_argument("--data", type=str, default="config/layout_data.yaml", help="Path to layout_data.yaml.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model (yolov8n.pt, yolov8s.pt, yolo11n.pt).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (reduce if VRAM is constrained).")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size (1024 recommended for documents).")
    parser.add_argument("--device", type=str, default="0", help="Device to use ('0' for GPU, 'cpu' for CPU).")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    args = parser.parse_args()
    
    train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        patience=args.patience
    )
