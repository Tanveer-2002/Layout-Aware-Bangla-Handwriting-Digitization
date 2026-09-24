"""
Region of Interest (RoI) Crop Exporter (Step 02 -> Step 03 Bridge)
Crops detected text regions from page scans with padding (to prevent Matra clipping)
and prepares them directly for the downstream Bangla TrOCR sequence recognizer.
"""

import sys
import cv2
import json
import argparse
from pathlib import Path
from typing import Union

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from src.utils.preprocess import preprocess_document
    from src.utils.spatial_json import format_layout_json, save_layout_json
except ModuleNotFoundError:
    from utils.preprocess import preprocess_document
    from utils.spatial_json import format_layout_json, save_layout_json

from ultralytics import YOLO


def export_text_rois(
    image_path: str,
    model_path: str = "yolov8n.pt",
    output_crops_dir: str = "crops",
    padding_px: int = 10,
    conf_thresh: float = 0.25,
    device: str = "0"
):
    """
    Detects text regions, cuts out cropped line/paragraph images with dilation padding,
    and indexes them in a JSON manifest for TrOCR.
    """
    img_path = Path(image_path)
    crops_root = Path(output_crops_dir)
    crops_root.mkdir(parents=True, exist_ok=True)
    
    # Load model with fallback if weights/best_layout.pt is not yet trained
    model_file = Path(model_path)
    if not model_file.exists():
        runs_best = list(Path("runs/train").glob("**/weights/best.pt"))
        if runs_best:
            runs_best.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            model_file = runs_best[0]
            print(f"[INFO] Using latest trained weights found at: {model_file}")
        elif Path("yolov8n.pt").exists():
            print("\n" + "!" * 65)
            print(f"[WARNING] Model weights file not found: '{model_path}'")
            print("Falling back to base pretrained model: 'yolov8n.pt'")
            print("Note: To get accurate Bangla text line detections, train the model first:")
            print("  python src/train.py --epochs 30 --batch 8 --device 0")
            print("!" * 65 + "\n")
            model_file = Path("yolov8n.pt")
        else:
            raise FileNotFoundError(f"Model weights not found at: {model_path}")
            
    print(f"Loading YOLO model: {model_file}")
    model = YOLO(str(model_file))
    proc_img, meta = preprocess_document(img_path, deskew=True, enhance=True)
    h, w = proc_img.shape[:2]
    
    preds = model.predict(source=proc_img, conf=conf_thresh, device=device, verbose=False)[0]
    
    raw_boxes = []
    names = model.names
    for box in preds.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        raw_boxes.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": cls_id,
            "class_name": names.get(cls_id, f"class_{cls_id}")
        })
        
    layout_data = format_layout_json(img_path.stem, [w, h], raw_boxes, meta)
    
    # Save crops in reading order
    doc_crops_dir = crops_root / img_path.stem
    doc_crops_dir.mkdir(parents=True, exist_ok=True)
    
    crop_manifest = []
    for r in layout_data["regions"]:
        order = r["reading_order"]
        x1, y1, x2, y2 = map(int, r["bbox"])
        
        # Apply padding (safety margin for Bengali ascenders/descenders/matra)
        x1_pad = max(0, x1 - padding_px)
        y1_pad = max(0, y1 - padding_px)
        x2_pad = min(w, x2 + padding_px)
        y2_pad = min(h, y2 + padding_px)
        
        crop = proc_img[y1_pad:y2_pad, x1_pad:x2_pad]
        crop_filename = f"{img_path.stem}_region_{order:03d}.jpg"
        crop_path = doc_crops_dir / crop_filename
        cv2.imwrite(str(crop_path), crop)
        
        crop_manifest.append({
            "region_id": order,
            "crop_file": str(crop_path.relative_to(crops_root)),
            "bbox_padded": [x1_pad, y1_pad, x2_pad, y2_pad],
            "bbox_original": [x1, y1, x2, y2],
            "confidence": r["confidence"],
            "class_name": r["class_name"]
        })
        
    manifest_path = doc_crops_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "document": img_path.name,
            "total_crops": len(crop_manifest),
            "crops": crop_manifest
        }, f, indent=2)
        
    print(f"\nSuccessfully exported {len(crop_manifest)} text region crops to: {doc_crops_dir}")
    print(f"Crop manifest: {manifest_path}")
    return doc_crops_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export cropped text regions for TrOCR.")
    parser.add_argument("--image", type=str, required=True, help="Input document image.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO weights.")
    parser.add_argument("--output-dir", type=str, default="crops", help="Directory for cropped regions.")
    parser.add_argument("--padding", type=int, default=10, help="Padding in pixels around crop.")
    parser.add_argument("--device", type=str, default="0", help="Device ('0' or 'cpu').")
    args = parser.parse_args()
    
    export_text_rois(args.image, args.model, args.output_dir, args.padding, device=args.device)
