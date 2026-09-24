"""
Inference Script for Document Layout Detection
Performs end-to-end layout analysis on full document scans:
1. Optional OpenCV preprocessing (deskew + contrast enhancement)
2. YOLO layout detection
3. Visual bounding box annotation
4. Structured spatial JSON export
"""

import sys
import argparse
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from src.utils.preprocess import preprocess_document
    from src.utils.spatial_json import format_layout_json, save_layout_json
    from src.utils.visualizer import draw_layout_boxes, save_visualization
except ModuleNotFoundError:
    from utils.preprocess import preprocess_document
    from utils.spatial_json import format_layout_json, save_layout_json
    from utils.visualizer import draw_layout_boxes, save_visualization

from ultralytics import YOLO


def predict_layout(
    image_path: str,
    model_path: str = "yolov8n.pt",
    output_dir: str = "outputs",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
    img_size: int = 1024,
    device: str = "0",
    apply_preprocess: bool = True
):
    """
    Runs layout detection pipeline on a single document scan or an entire directory.
    """
    img_path = Path(image_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
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
    
    # Determine input files
    if img_path.is_dir():
        image_files = [f for f in img_path.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    else:
        image_files = [img_path]
        
    print(f"Processing {len(image_files)} image(s)...")
    
    results = []
    for img_file in image_files:
        doc_name = img_file.stem
        print(f"\nDetecting layout: {img_file.name}")
        
        # Step 1: Preprocessing
        if apply_preprocess:
            proc_img, meta = preprocess_document(img_file, deskew=True, enhance=True)
        else:
            import cv2
            proc_img = cv2.imread(str(img_file))
            meta = {"original_size": [proc_img.shape[1], proc_img.shape[0]]}
            
        h, w = proc_img.shape[:2]
        
        # Step 2: YOLO Detection
        preds = model.predict(
            source=proc_img,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=img_size,
            device=device,
            verbose=False
        )[0]
        
        # Extract detected boxes
        boxes_data = []
        names = model.names
        for box in preds.boxes:
            # Absolute coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, f"class_{cls_id}")
            
            # Normalized coordinates [xc, yc, w, h]
            xc, yc, bw, bh = box.xywhn[0].tolist()
            
            boxes_data.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "bbox_normalized": [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)],
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "class_name": cls_name
            })
            
        # Step 3: Information Fusion (JSON Schema)
        layout_json = format_layout_json(doc_name, [w, h], boxes_data, meta)
        json_out = out_root / f"{doc_name}_layout.json"
        save_layout_json(layout_json, json_out)
        print(f"  -> Layout JSON: {json_out} ({layout_json['total_regions']} regions)")
        
        # Step 4: Visualization
        vis_img = draw_layout_boxes(proc_img, layout_json["regions"])
        vis_out = out_root / f"{doc_name}_annotated.jpg"
        save_visualization(vis_img, vis_out)
        print(f"  -> Annotated Image: {vis_out}")
        
        results.append((json_out, vis_out))
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO layout prediction on documents.")
    parser.add_argument("--image", type=str, required=True, help="Path to document image or folder.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model weights.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--device", type=str, default="0", help="Device ('0' or 'cpu').")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable OpenCV deskew/contrast.")
    args = parser.parse_args()
    
    predict_layout(
        image_path=args.image,
        model_path=args.model,
        output_dir=args.output_dir,
        conf_thresh=args.conf,
        img_size=args.imgsz,
        device=args.device,
        apply_preprocess=not args.no_preprocess
    )
