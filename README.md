# Layout-Aware Bangla Handwriting Digitization

End-to-end framework converting physical handwritten Bangla document archives into editable, layout-preserved digital PDF and DOCX files.

This repository contains the **Document Layout Detection Subsystem (Step 02)** powered by **Ultralytics YOLO**, with integrated OpenCV preprocessing and Region-of-Interest (RoI) crop generation for downstream **Bangla TrOCR (Step 03)** and **Layout Reconstruction (Step 04 & 05)**.

---

## Repository Structure

```
Layout-Aware-Bangla-Handwriting-Digitization/
├── .gitignore                      # Excludes raw data, weights, runs, and cache
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── config/
│   └── layout_data.yaml            # YOLO dataset configuration
├── src/
│   ├── __init__.py
│   ├── dataset_prep.py             # Prepares BN-HTRd dataset into YOLO train/val/test splits
│   ├── train.py                    # Script to train YOLO layout detector
│   ├── evaluate.py                 # Evaluates model metrics (mAP@50, mAP@50-95, precision, recall)
│   ├── predict.py                  # Full-page layout inference + JSON + visualization
│   ├── export_rois.py              # Crops text regions with padding for TrOCR
│   └── utils/
│       ├── __init__.py
│       ├── preprocess.py           # OpenCV deskewing (Hough) & adaptive contrast
│       ├── spatial_json.py         # Intermediate spatial JSON schema & reading order
│       └── visualizer.py           # Annotates bounding boxes and reading order paths
└── scripts/
    ├── run_prep.py                 # One-click dataset preparation script
    ├── run_train.py                # One-click YOLO training script
    └── run_inference.py            # Quick test prediction script
```

---

## 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/MD-Afnan10/Layout-Aware-Bangla-Handwriting-Digitization.git
cd Layout-Aware-Bangla-Handwriting-Digitization

pip install -r requirements.txt
```

---

## 2. Dataset Preparation

The dataset converter prepares the raw **BN-HTRd** dataset (`Dataset/` containing folders `1` to `150`) into standard YOLO detection format:
- Groups pages by document to **prevent data leakage** between splits.
- Formats normalized bounding boxes for text lines.
- Splits into `train` (80%), `val` (10%), and `test` (10%).

Run the preparation script:

```bash
python scripts/run_prep.py
```

Or configure custom paths:

```bash
python src/dataset_prep.py --dataset-dir "f:/DIP/Dataset" --output-dir "data/yolo_dataset" --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

> **Note:** The `data/` and `Dataset/` directories are automatically ignored by `.gitignore` and will never be pushed to git.

---

## 3. Training YOLO Layout Detector

Train a high-resolution YOLO model (1024px image size is recommended for capturing thin handwritten text baselines):

```bash
python src/train.py --data config/layout_data.yaml --model yolov8n.pt --epochs 50 --batch 8 --imgsz 1024 --device 0
```

Trained weights are automatically saved to `weights/best_layout.pt`.

---

## 4. Evaluation & Metrics

Evaluate the model against the validation or test split:

```bash
python src/evaluate.py --model weights/best_layout.pt --data config/layout_data.yaml --split val
```

Outputs:
- **mAP@50** and **mAP@50-95**
- **Precision & Recall**
- **Intersection over Union (IoU)**

---

## 5. Inference & Spatial JSON Generation

Run layout detection on a document page to generate:
1. Annotated visualization image (`*_annotated.jpg`)
2. Structured layout metadata JSON (`*_layout.json`) with reading order and pixel coordinates:

```bash
python src/predict.py --image "f:/DIP/Dataset/1/1_1.jpg" --model weights/best_layout.pt --output-dir outputs/
```

Sample JSON output:
```json
{
  "document_name": "1_1",
  "page_dimensions": {"width": 2288, "height": 2196},
  "total_regions": 12,
  "regions": [
    {
      "reading_order": 1,
      "class_name": "text_line",
      "confidence": 0.94,
      "bbox": [795.0, 102.0, 1290.0, 183.0],
      "bbox_normalized": [0.4556, 0.0649, 0.2163, 0.0369]
    }
  ]
}
```

---

## 6. Exporting RoI Crops for TrOCR (Step 02 → Step 03)

To crop individual text regions with safety dilation padding (preventing clipping of Matra and vowel modifiers) for input into the **Bangla TrOCR model**:

```bash
python src/export_rois.py --image "f:/DIP/Dataset/1/1_1.jpg" --model weights/best_layout.pt --output-dir crops/
```

---

## Git Policy

All large datasets (`data/`, `Dataset/`), model weights (`*.pt`, `weights/`), training logs (`runs/`), and generated outputs (`outputs/`, `crops/`) are strictly ignored via `.gitignore` to keep the main branch clean and lightweight.
