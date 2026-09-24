"""Convenience script to launch YOLO training"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.train import train_yolo

if __name__ == "__main__":
    data_yaml = str(repo_root / "data" / "yolo_dataset" / "layout_data.yaml")
    if not Path(data_yaml).exists():
        data_yaml = str(repo_root / "config" / "layout_data.yaml")
        
    train_yolo(data_yaml=data_yaml, model_name="yolov8n.pt", epochs=30, batch_size=8, img_size=1024)
