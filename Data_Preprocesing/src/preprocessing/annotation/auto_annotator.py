"""
Auto-Annotation Fallback Module for Grape Bunch Detection.
When external annotation files (.xml, .json, .txt) are absent, this module uses
computer vision (HSV/LAB color segmentation + saliency/contour detection) to detect
grape bunch regions and auto-generate normalized YOLO bounding box annotations.
"""

import cv2
import numpy as np
from typing import List
from src.preprocessing.contracts import BoundingBox


def detect_grape_bunches_auto(image_path: str, min_area_ratio: float = 0.005, max_area_ratio: float = 0.90) -> List[BoundingBox]:
    """
    Detects grape bunch regions in an image using multi-color space saliency and contour analysis.
    Returns a list of BoundingBox objects normalized to [0, 1].
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    img_area = float(w * h)

    # Convert to HSV and LAB color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 1. Color Segmentation for Green / Yellow-Green Grapes
    lower_green = np.array([25, 35, 35])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 2. Color Segmentation for Purple / Dark Red Grapes
    lower_purple1 = np.array([125, 30, 30])
    upper_purple1 = np.array([175, 255, 255])
    lower_purple2 = np.array([0, 30, 30])
    upper_purple2 = np.array([15, 255, 255])
    mask_purple = cv2.bitwise_or(
        cv2.inRange(hsv, lower_purple1, upper_purple1),
        cv2.inRange(hsv, lower_purple2, upper_purple2)
    )

    # 3. LAB Saliency Mask (for high contrast grape cluster regions against artificial/coral/vineyard backgrounds)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    _, mask_saliency = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combined mask
    combined_mask = cv2.bitwise_or(mask_green, mask_purple)
    if cv2.countNonZero(combined_mask) < 0.01 * img_area:
        # Fallback to saliency/contrast mask if specific color masks are sparse
        combined_mask = cv2.bitwise_or(combined_mask, cv2.bitwise_and(mask_saliency, cv2.bitwise_not(mask_green)))

    # Morphological cleaning
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_open)

    # Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[BoundingBox] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / img_area
        if min_area_ratio <= area_ratio <= max_area_ratio:
            bx, by, bw, bh = cv2.boundingRect(cnt)

            # Normalize to YOLO format (0.0 to 1.0)
            xc = float(bx + bw / 2.0) / w
            yc = float(by + bh / 2.0) / h
            norm_w = float(bw) / w
            norm_h = float(bh) / h

            # Clip within [0, 1]
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            box = BoundingBox(
                class_id=0,
                x_center=xc,
                y_center=yc,
                width=norm_w,
                height=norm_h,
            )
            boxes.append(box)

    # Fallback: If no contours met criteria (or background was uniform/difficult), generate central ROI cluster fallback box
    if not boxes:
        # Default central 60% ROI bounding box representing single cluster
        boxes.append(BoundingBox(
            class_id=0,
            x_center=0.5,
            y_center=0.5,
            width=0.6,
            height=0.6,
        ))

    return boxes
