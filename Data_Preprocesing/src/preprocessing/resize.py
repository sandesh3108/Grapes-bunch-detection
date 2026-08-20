"""
Stage 6 — Aspect-Ratio Preserving Letterbox Resize Module.
Resizes images to target resolution (640x640) with letterbox padding and transforms bounding box coordinates identically.
"""

import cv2
import numpy as np
from typing import Tuple, List
from src.preprocessing.contracts import BoundingBox
from src.preprocessing.annotation.validator import sanitize_bounding_box


def letterbox_image(
    image: np.ndarray,
    target_size: int = 640,
    padding_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, float, float]:
    """
    Resizes image maintaining aspect ratio and adds letterbox padding.
    Returns (letterboxed_image, scale_factor, pad_w_pixels, pad_h_pixels).
    """
    orig_h, orig_w = image.shape[:2]
    scale = min(target_size / orig_w, target_size / orig_h)

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) / 2.0
    pad_h = (target_size - new_h) / 2.0

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))

    canvas = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color
    )

    # Ensure canvas is exact target size
    if canvas.shape[0] != target_size or canvas.shape[1] != target_size:
        canvas = cv2.resize(canvas, (target_size, target_size))

    return canvas, scale, pad_w, pad_h


def transform_bbox_letterbox(
    box: BoundingBox,
    orig_w: int,
    orig_h: int,
    target_size: int,
    scale: float,
    pad_w: float,
    pad_h: float,
) -> BoundingBox:
    """
    Transforms normalized bounding box from original image dimensions to letterboxed canvas dimensions.
    """
    # 1. Un-normalize to original pixel coordinates
    bw_pixels = box.width * orig_w
    bh_pixels = box.height * orig_h
    bx_pixels = box.x_center * orig_w
    by_pixels = box.y_center * orig_h

    # 2. Scale and shift pixel coordinates
    new_bx_pixels = bx_pixels * scale + pad_w
    new_by_pixels = by_pixels * scale + pad_h
    new_bw_pixels = bw_pixels * scale
    new_bh_pixels = bh_pixels * scale

    # 3. Re-normalize to target_size canvas
    new_xc = new_bx_pixels / target_size
    new_yc = new_by_pixels / target_size
    new_w = new_bw_pixels / target_size
    new_h = new_bh_pixels / target_size

    new_box = BoundingBox(
        class_id=box.class_id,
        x_center=new_xc,
        y_center=new_yc,
        width=new_w,
        height=new_h,
    )
    return sanitize_bounding_box(new_box)
