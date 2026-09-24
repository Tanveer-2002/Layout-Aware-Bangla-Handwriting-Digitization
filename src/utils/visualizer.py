"""
Document Layout Visualization Engine
Renders detection bounding boxes, confidence badges, reading order numbers,
and optional reading path connections.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union


# High-contrast color palette for layout classes
CLASS_COLORS = {
    0: (235, 132, 56),    # Sky Blue (BGR: 56, 189, 248 -> BGR is (248, 189, 56))
    1: (168, 85, 247),    # Purple
    2: (16, 185, 129),    # Emerald
    "default": (244, 63, 94) # Rose
}


def draw_layout_boxes(
    image: np.ndarray,
    regions: List[Dict[str, Any]],
    draw_order_path: bool = True,
    draw_badges: bool = True
) -> np.ndarray:
    """
    Draws semi-transparent bounding boxes and reading order badges on the image.
    """
    vis = image.copy()
    overlay = vis.copy()
    h, w = vis.shape[:2]
    
    # First pass: filled semi-transparent boxes
    for r in regions:
        bbox = r["bbox"]  # [xmin, ymin, xmax, ymax] in pixels
        x1, y1, x2, y2 = map(int, bbox)
        cls_id = r.get("class_id", 0)
        color = CLASS_COLORS.get(cls_id, CLASS_COLORS["default"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
    # Blend overlay with 15% opacity
    cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
    
    # Second pass: borders and badges
    centers = []
    for r in regions:
        bbox = r["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        cls_id = r.get("class_id", 0)
        cls_name = r.get("class_name", f"class_{cls_id}")
        conf = r.get("confidence", 1.0)
        order = r.get("reading_order", 0)
        color = CLASS_COLORS.get(cls_id, CLASS_COLORS["default"])
        
        # Border
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # Center for reading path
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        
        if draw_badges:
            # Order badge on top-left of box
            badge_text = f"#{order} {cls_name} {conf:.2f}" if conf < 1.0 else f"#{order} {cls_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(badge_text, font, scale, thickness)
            
            # Badge background
            by1 = max(0, y1 - th - 6)
            by2 = y1
            bx1 = x1
            bx2 = min(w, x1 + tw + 10)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, -1)
            cv2.putText(vis, badge_text, (bx1 + 5, by2 - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Optional: Draw reading path line connecting consecutive regions
    if draw_order_path and len(centers) > 1:
        for i in range(len(centers) - 1):
            pt1 = centers[i]
            pt2 = centers[i + 1]
            cv2.arrowedLine(vis, pt1, pt2, (100, 116, 139), 1, tipLength=0.03, line_type=cv2.LINE_AA)
            
    return vis


def save_visualization(image: np.ndarray, output_path: Union[str, Path]):
    """Saves visualization image to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
