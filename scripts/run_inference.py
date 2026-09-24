"""Convenience script to run inference on a sample image"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.predict import predict_layout

if __name__ == "__main__":
    # Test on a sample from Dataset/1/1_1.jpg if available
    sample_img = Path("f:/DIP/Dataset/1/1_1.jpg")
    if not sample_img.exists():
        print("Please provide a valid image path to run inference.")
        sys.exit(1)
        
    predict_layout(
        image_path=str(sample_img),
        model_path="yolov8n.pt",
        output_dir=str(repo_root / "outputs"),
        conf_thresh=0.25
    )
