"""
Annotation and Bounding Box Validator.
Validates bbox values, normalized coordinates, and class IDs against dataset constraints.
"""

from typing import List, Tuple, Optional, Dict
from src.preprocessing.contracts import BoundingBox


def validate_bounding_box(
    box: BoundingBox,
    allowed_classes: Dict[int, str],
    min_area_ratio: float = 0.00001,
    max_area_ratio: float = 1.0,
    tolerance: float = 1e-3,
) -> Tuple[bool, Optional[str]]:
    """
    Validates a single bounding box.
    Returns (is_valid, error_message).
    """
    if box.class_id not in allowed_classes:
        return False, f"Invalid class_id {box.class_id}. Allowed: {list(allowed_classes.keys())}"

    if not (0.0 - tolerance <= box.x_center <= 1.0 + tolerance):
        return False, f"x_center {box.x_center} out of bounds [0, 1]"
    if not (0.0 - tolerance <= box.y_center <= 1.0 + tolerance):
        return False, f"y_center {box.y_center} out of bounds [0, 1]"

    if box.width <= 0.0 or box.height <= 0.0:
        return False, f"Non-positive dimensions width={box.width}, height={box.height}"

    if box.width > 1.0 + tolerance or box.height > 1.0 + tolerance:
        return False, f"Dimensions width={box.width}, height={box.height} exceed 1.0"

    area_ratio = box.width * box.height
    if area_ratio < min_area_ratio:
        return False, f"Bounding box area ratio {area_ratio:.6f} below minimum {min_area_ratio}"
    if area_ratio > max_area_ratio:
        return False, f"Bounding box area ratio {area_ratio:.6f} exceeds maximum {max_area_ratio}"

    x_min = box.x_center - box.width / 2.0
    x_max = box.x_center + box.width / 2.0
    y_min = box.y_center - box.height / 2.0
    y_max = box.y_center + box.height / 2.0

    if x_min < 0.0 - tolerance or x_max > 1.0 + tolerance or y_min < 0.0 - tolerance or y_max > 1.0 + tolerance:
        return False, f"Box edges [{x_min:.4f}, {x_max:.4f}, {y_min:.4f}, {y_max:.4f}] extend beyond image boundary [0, 1]"

    return True, None


def sanitize_bounding_box(box: BoundingBox) -> BoundingBox:
    """Clamps bounding box coordinates to strict [0, 1] range."""
    x_center = max(0.0, min(1.0, box.x_center))
    y_center = max(0.0, min(1.0, box.y_center))
    width = max(0.0, min(1.0, box.width))
    height = max(0.0, min(1.0, box.height))
    return BoundingBox(
        class_id=box.class_id,
        x_center=round(x_center, 6),
        y_center=round(y_center, 6),
        width=round(width, 6),
        height=round(height, 6),
    )
