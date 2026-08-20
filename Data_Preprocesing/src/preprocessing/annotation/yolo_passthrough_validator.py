"""
YOLO Format Passthrough Validator.
Parses existing YOLO txt annotation files, validates bounding boxes, and returns AnnotationRecord.
"""

from pathlib import Path
from typing import Dict, List
from src.preprocessing.contracts import AnnotationRecord, BoundingBox
from src.preprocessing.annotation.validator import validate_bounding_box, sanitize_bounding_box


def parse_yolo_txt(
    txt_path: str,
    image_path: str,
    allowed_classes: Dict[int, str],
) -> AnnotationRecord:
    """Parses and validates a YOLO txt file."""
    path = Path(txt_path)
    if not path.exists():
        return AnnotationRecord(
            image_path=image_path,
            annotation_path=txt_path,
            source_format="yolo_txt",
            valid=False,
            error_message=f"YOLO txt file not found: {txt_path}",
        )

    boxes: List[BoundingBox] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line_num, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            box = sanitize_bounding_box(
                BoundingBox(
                    class_id=cls_id,
                    x_center=xc,
                    y_center=yc,
                    width=w,
                    height=h,
                )
            )

            is_valid, err_msg = validate_bounding_box(box, allowed_classes)
            if is_valid:
                boxes.append(box)
    except Exception as e:
        return AnnotationRecord(
            image_path=image_path,
            annotation_path=txt_path,
            source_format="yolo_txt",
            valid=False,
            error_message=f"Error reading YOLO txt file: {str(e)}",
        )

    return AnnotationRecord(
        image_path=image_path,
        annotation_path=txt_path,
        source_format="yolo_txt",
        boxes=boxes,
        valid=True,
    )
