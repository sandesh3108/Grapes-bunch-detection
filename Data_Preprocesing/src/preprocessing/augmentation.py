"""
Stage 8 — Train-Only Augmentation Module.
Applies domain-specific augmentations (flips, rotation, HSV jitter, brightness/contrast, occlusion, motion blur)
ONLY to training images, with bounding-box aware coordinate transforms.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from src.preprocessing.contracts import BoundingBox
from src.preprocessing.annotation.validator import sanitize_bounding_box

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def build_albumentations_pipeline(aug_config: Dict[str, Any]) -> Any:
    """Builds an Albumentations Compose pipeline with yolo bbox format support."""
    transforms = []

    hflip_cfg = aug_config.get("horizontal_flip", {})
    if hflip_cfg.get("enabled", True):
        transforms.append(A.HorizontalFlip(p=float(hflip_cfg.get("probability", 0.5))))

    rot_cfg = aug_config.get("rotation", {})
    if rot_cfg.get("enabled", True):
        deg = float(rot_cfg.get("degrees", 15))
        transforms.append(A.Rotate(limit=(-deg, deg), border_mode=cv2.BORDER_CONSTANT, value=114, p=float(rot_cfg.get("probability", 0.5))))

    hsv_cfg = aug_config.get("hsv", {})
    if hsv_cfg.get("enabled", True):
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=int(hsv_cfg.get("hue_shift_limit", 10)),
                sat_shift_limit=int(hsv_cfg.get("sat_shift_limit", 20)),
                val_shift_limit=int(hsv_cfg.get("val_shift_limit", 20)),
                p=float(hsv_cfg.get("probability", 0.5)),
            )
        )

    bc_cfg = aug_config.get("brightness_contrast", {})
    if bc_cfg.get("enabled", True):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(bc_cfg.get("brightness_limit", 0.2)),
                contrast_limit=float(bc_cfg.get("contrast_limit", 0.2)),
                p=float(bc_cfg.get("probability", 0.5)),
            )
        )

    mblur_cfg = aug_config.get("motion_blur", {})
    if mblur_cfg.get("enabled", True):
        transforms.append(
            A.MotionBlur(
                blur_limit=int(mblur_cfg.get("blur_limit", 7)),
                p=float(mblur_cfg.get("probability", 0.3)),
            )
        )

    occ_cfg = aug_config.get("occlusion_simulation", {})
    if occ_cfg.get("enabled", True):
        max_h = int(occ_cfg.get("max_h_size", 32))
        max_w = int(occ_cfg.get("max_w_size", 32))
        num_h = int(occ_cfg.get("num_holes", 4))
        try:
            dropout = A.CoarseDropout(
                num_holes_range=(1, num_h),
                hole_height_range=(8, max_h),
                hole_width_range=(8, max_w),
                fill_value=114,
                p=float(occ_cfg.get("probability", 0.3)),
            )
        except Exception:
            dropout = A.CoarseDropout(
                max_holes=num_h,
                max_height=max_h,
                max_width=max_w,
                fill_value=114,
                p=float(occ_cfg.get("probability", 0.3)),
            )
        transforms.append(dropout)

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2),
    )


def augment_training_sample(
    image: np.ndarray,
    boxes: List[BoundingBox],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, List[BoundingBox]]:
    """
    Applies augmentations to a single training image and its bounding boxes.
    """
    if not boxes or not ALBUMENTATIONS_AVAILABLE:
        return image, boxes

    aug_config = config.get("augmentation", {})
    pipeline = build_albumentations_pipeline(aug_config)

    albu_bboxes = []
    class_labels = []
    for box in boxes:
        albu_bboxes.append([box.x_center, box.y_center, box.width, box.height])
        class_labels.append(box.class_id)

    try:
        augmented = pipeline(image=image, bboxes=albu_bboxes, class_labels=class_labels)
        aug_img = augmented["image"]
        aug_boxes_raw = augmented["bboxes"]
        aug_labels = augmented["class_labels"]

        aug_boxes: List[BoundingBox] = []
        for b, cls_id in zip(aug_boxes_raw, aug_labels):
            box = sanitize_bounding_box(
                BoundingBox(
                    class_id=cls_id,
                    x_center=b[0],
                    y_center=b[1],
                    width=b[2],
                    height=b[3],
                )
            )
            aug_boxes.append(box)

        return aug_img, aug_boxes
    except Exception:
        # Fallback to original image if augmentation fails
        return image, boxes
