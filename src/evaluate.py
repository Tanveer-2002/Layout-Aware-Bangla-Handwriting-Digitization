"""
Evaluation Script for YOLO Layout Detection
Evaluates model on validation/test split and computes:
- Precision, Recall, mAP@0.5, mAP@0.5:0.95
- IoU overlap metrics
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO


def evaluate_model(
    model_path: str = "weights/best_layout.pt",
    data_yaml: str = "config/layout_data.yaml",
    img_size: int = 1024,
    device: str = "0",
    split: str = "val"
):
    """Evaluates trained YOLO layout detector."""
    model_file = Path(model_path)
    if not model_file.exists():
        # Check if there are any weights in runs/train
        runs_best = list(Path("runs/train").glob("**/weights/best.pt"))
        if runs_best:
            # Pick latest modified best.pt
            runs_best.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            model_file = runs_best[0]
            print(f"[INFO] Using latest trained weights found at: {model_file}")
        else:
            print("\n" + "!" * 65)
            print(f"[ERROR] Model weights file not found: '{model_path}'")
            print("!" * 65)
            print("Reason: 'weights/best_layout.pt' is generated after training.")
            print("You have not trained the model yet.\n")
            print("To train the YOLO layout model:")
            print("  python src/train.py --epochs 30 --batch 8 --device 0")
            print("  or:")
            print("  python scripts/run_train.py\n")
            print("To test the evaluation pipeline with the base pretrained model:")
            print(f"  python src/evaluate.py --model yolov8n.pt --split {split}")
            print("!" * 65 + "\n")
            sys.exit(1)
            
    print(f"Loading model: {model_file}")
    model = YOLO(str(model_file))
    
    print(f"Evaluating on split: '{split}' (imgsz={img_size})...")
    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        device=device,
        split=split,
        plots=True
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS:")
    print("="*50)
    print(f"mAP@50       : {metrics.box.map50:.4f}")
    print(f"mAP@50-95    : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print("="*50)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO layout detector.")
    parser.add_argument("--model", type=str, default="weights/best_layout.pt", help="Path to trained .pt weights.")
    parser.add_argument("--data", type=str, default="config/layout_data.yaml", help="Path to layout_data.yaml.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Evaluation image size.")
    parser.add_argument("--device", type=str, default="0", help="Device ('0' or 'cpu').")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split.")
    args = parser.parse_args()
    
    evaluate_model(args.model, args.data, args.imgsz, args.device, args.split)
