"""Convenience script to prepare dataset from f:/DIP/Dataset"""
import sys
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.dataset_prep import prepare_yolo_dataset

if __name__ == "__main__":
    dataset_dir = "f:/DIP/Dataset"
    output_dir = str(repo_root / "data" / "yolo_dataset")
    prepare_yolo_dataset(dataset_dir, output_dir)
