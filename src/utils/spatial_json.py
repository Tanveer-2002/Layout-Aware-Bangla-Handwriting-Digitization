"""
Spatial JSON Formatter (Step 04 in Project Architecture)
Binds YOLO bounding boxes with metadata, calculates natural reading order
(top-to-bottom, left-to-right), and generates the intermediate JSON schema.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union


def sort_reading_order(boxes: List[Dict[str, Any]], line_tolerance: float = 0.02) -> List[Dict[str, Any]]:
    """
    Sorts bounding boxes into natural document reading order:
    Primary: Top-to-bottom (y-axis)
    Secondary: Left-to-right (x-axis) within the same line threshold.
    
    Args:
        boxes: List of dicts with 'bbox_normalized' [ymin, xmin, ymax, xmax] or 'box_2d'
        line_tolerance: Normalized vertical tolerance to group boxes onto the same line.
    """
    if not boxes:
        return []

    # Sort primarily by ymin
    sorted_by_y = sorted(boxes, key=lambda b: b["bbox"][1])
    
    lines = []
    current_line = [sorted_by_y[0]]
    
    for b in sorted_by_y[1:]:
        # Compare ymin of current box with median ymin of the current line
        line_y = sum(x["bbox"][1] for x in current_line) / len(current_line)
        line_height = sum(x["bbox"][3] - x["bbox"][1] for x in current_line) / len(current_line)
        tolerance = max(line_tolerance, line_height * 0.4)
        
        if abs(b["bbox"][1] - line_y) <= tolerance:
            current_line.append(b)
        else:
            # Sort current line left-to-right
            current_line.sort(key=lambda x: x["bbox"][0])
            lines.extend(current_line)
            current_line = [b]
            
    if current_line:
        current_line.sort(key=lambda x: x["bbox"][0])
        lines.extend(current_line)
        
    # Re-assign reading order indices
    for idx, item in enumerate(lines, start=1):
        item["reading_order"] = idx
        
    return lines


def format_layout_json(
    document_name: str,
    page_dimensions: List[int],
    detections: List[Dict[str, Any]],
    preprocessing_meta: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generates structured intermediate layout schema for downstream TrOCR and export.
    """
    w, h = page_dimensions
    
    # Sort into reading order
    ordered_regions = sort_reading_order(detections)
    
    output = {
        "document_name": document_name,
        "page_dimensions": {
            "width": w,
            "height": h
        },
        "preprocessing": preprocessing_meta or {},
        "total_regions": len(ordered_regions),
        "regions": ordered_regions
    }
    
    return output


def save_layout_json(data: Dict[str, Any], output_path: Union[str, Path]):
    """Saves structured JSON to disk with clean formatting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
