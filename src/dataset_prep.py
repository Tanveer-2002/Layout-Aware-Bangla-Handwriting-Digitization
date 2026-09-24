"""
Dataset Preparation for YOLO Layout Detection
Converts BN-HTRd dataset (or custom layout scans) into standard YOLO detection format:
- Splits by document/folder into train / val / test sets (prevents data leakage)
- Re-indexes class 1 (line) to 0 (text_line)
- Validates bounding box coordinates
- Generates data/layout_data.yaml
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm


def collect_document_samples(dataset_root: Path) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Scans the raw BN-HTRd dataset directory.
    Returns a dict mapping doc_id -> list of (image_path, annotation_txt_path).
    """
    doc_samples = {}
    
    # Subfolders are numbered 1 to 150
    subdirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    subdirs.sort(key=lambda x: int(x.name) if x.name.isdigit() else str(x.name))
    
    for doc_dir in subdirs:
        doc_id = doc_dir.name
        lines_dir = doc_dir / "Lines"
        
        # Find all page scans (e.g. 1_1.jpg, 1_2.jpg, 1_3.jpg)
        page_images = [
            f for f in doc_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        page_images.sort(key=lambda x: x.stem)
        
        pairs = []
        for img_path in page_images:
            txt_path = lines_dir / f"{img_path.stem}.txt"
            if txt_path.exists():
                pairs.append((img_path, txt_path))
                
        if pairs:
            doc_samples[doc_id] = pairs
            
    return doc_samples


def convert_and_write_label(src_txt: Path, dst_txt: Path, target_class_id: int = 0):
    """
    Reads source bounding boxes, re-maps the class ID to target_class_id (default 0),
    clamps coordinates to [0, 1], and writes the sanitized YOLO label.
    """
    valid_lines = []
    with open(src_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Raw format: class_id x_center y_center width height
                try:
                    xc, yc, w, h = map(float, parts[1:5])
                    # Clamp to [0, 1]
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))
                    if w > 0.001 and h > 0.001:
                        valid_lines.append(f"{target_class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                except ValueError:
                    continue
                    
    with open(dst_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_lines) + "\n")


def prepare_yolo_dataset(
    raw_dataset_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy_files: bool = True
):
    """
    Prepares complete YOLO dataset directory structure.
    """
    raw_root = Path(raw_dataset_dir).resolve()
    out_root = Path(output_dir).resolve()
    
    print(f"Scanning dataset in: {raw_root}")
    doc_samples = collect_document_samples(raw_root)
    total_docs = len(doc_samples)
    total_pages = sum(len(v) for v in doc_samples.values())
    print(f"Found {total_docs} documents with {total_pages} total page scans.")
    
    if total_docs == 0:
        raise ValueError(f"No valid document/page pairs found in {raw_root}")
        
    # Split by document ID to avoid data leakage
    doc_ids = list(doc_samples.keys())
    random.seed(seed)
    random.shuffle(doc_ids)
    
    n_train = int(total_docs * train_ratio)
    n_val = int(total_docs * val_ratio)
    
    train_docs = set(doc_ids[:n_train])
    val_docs = set(doc_ids[n_train:n_train + n_val])
    test_docs = set(doc_ids[n_train + n_val:])
    
    splits = {
        "train": [pair for d in train_docs for pair in doc_samples[d]],
        "val": [pair for d in val_docs for pair in doc_samples[d]],
        "test": [pair for d in test_docs for pair in doc_samples[d]]
    }
    
    # Create directories
    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    print("\nGenerating YOLO dataset splits:")
    for split, pairs in splits.items():
        print(f"  Split '{split}': {len(pairs)} images")
        for img_path, txt_path in tqdm(pairs, desc=f"  Processing {split}"):
            dst_img = out_root / "images" / split / img_path.name
            dst_txt = out_root / "labels" / split / f"{img_path.stem}.txt"
            
            # Copy or link image
            if copy_files:
                shutil.copyfile(img_path, dst_img)
            else:
                try:
                    os.link(img_path, dst_img)
                except (OSError, AttributeError):
                    shutil.copyfile(img_path, dst_img)
                    
            # Convert and write label with class 0 (text_line)
            convert_and_write_label(txt_path, dst_txt, target_class_id=0)
            
    # Generate data yaml file
    yaml_content = f"""# YOLO Dataset Config for Bangla Layout Detection
path: {out_root.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: text_line
"""
    yaml_path = out_root / "layout_data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(f"\nDataset preparation complete! Saved to: {out_root}")
    print(f"Config YAML written to: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BN-HTRd dataset for YOLO layout detection.")
    parser.add_argument("--dataset-dir", type=str, default="f:/DIP/Dataset", help="Raw dataset root directory.")
    parser.add_argument("--output-dir", type=str, default="data/yolo_dataset", help="Target YOLO dataset directory.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    args = parser.parse_args()
    
    prepare_yolo_dataset(
        args.dataset_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
