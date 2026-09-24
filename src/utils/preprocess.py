"""
OpenCV Image Preprocessing Pipeline (Step 01 in Project Architecture)
Handles illumination normalization, contrast enhancement, median denoising,
and Hough transform deskewing.
"""

import cv2
import numpy as np
from typing import Tuple, Union
from pathlib import Path


def load_image(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Loads image from path or validates existing numpy array."""
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Failed to load image from: {image_input}")
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")


def estimate_skew_angle(image: np.ndarray, max_angle: float = 45.0) -> float:
    """
    Estimates the skew angle of a scanned document using Hough Line Transform
    and horizontal text baseline orientation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    # Adaptive thresholding or Otsu to get binary strokes
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=20
    )
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Keep only near-horizontal lines within max_angle
        if abs(angle) <= max_angle:
            angles.append(angle)
            
    if not angles:
        return 0.0
        
    # Return median angle to resist outliers
    median_angle = float(np.median(angles))
    return median_angle


def deskew_image(image: np.ndarray, angle: float = None) -> Tuple[np.ndarray, float]:
    """
    Straightens a skewed document image using affine rotation around the center.
    """
    if angle is None:
        angle = estimate_skew_angle(image)
        
    if abs(angle) < 0.1:
        return image, 0.0
        
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Compute affine transformation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Determine new bounding dimension to avoid clipping
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix with translation
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    
    # Fill border with white paper color
    border_color = (255, 255, 255) if len(image.shape) == 3 else 255
    deskewed = cv2.warpAffine(
        image,
        rot_mat,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color
    )
    
    return deskewed, angle


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Works on color (LAB space) or grayscale images.
    """
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(image)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Applies median and bilateral filtering to eliminate scan pepper noise."""
    if len(image.shape) == 3:
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        return cv2.medianBlur(image, 3)


def preprocess_document(
    image_input: Union[str, Path, np.ndarray],
    deskew: bool = True,
    enhance: bool = True,
    denoise: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline combining deskew, contrast normalization, and denoising.
    Returns:
        (preprocessed_image, metadata_dict)
    """
    img = load_image(image_input)
    h_orig, w_orig = img.shape[:2]
    skew_angle = 0.0
    
    if deskew:
        img, skew_angle = deskew_image(img)
        
    if enhance:
        img = enhance_contrast(img)
        
    if denoise:
        img = denoise_image(img)
        
    metadata = {
        "original_size": [w_orig, h_orig],
        "processed_size": [img.shape[1], img.shape[0]],
        "skew_angle_deg": round(skew_angle, 2)
    }
    
    return img, metadata
