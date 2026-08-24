"""
COCO JSON to YOLO Converter.
Parses COCO format JSON annotation files and converts bounding boxes to normalized YOLO format.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from src.preprocessing.contracts import AnnotationRecord, BoundingBox
from src.preprocessing.annotation.validator import validate_bounding_box, sanitize_bounding_box


def parse_coco_json(
    json_path: str,
    subfolder_path: str,
    class_map: Dict[str, int],
    allowed_classes: Dict[int, str],
) -> Dict[str, AnnotationRecord]:
    """
    Parses a COCO JSON file and returns a dictionary mapping image filenames/paths to AnnotationRecords.
    """
    path = Path(json_path)
    records: Dict[str, AnnotationRecord] = {}

    if not path.exists():
        return records

    try:
        with open(path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
    except Exception:
        return records

    images = {img["id"]: img for img in coco_data.get("images", [])}
    categories = {cat["id"]: cat.get("name", "grape_bunch") for cat in coco_data.get("categories", [])}

    # Group annotations by image_id
    annotations_by_img: Dict[int, List[dict]] = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        annotations_by_img.setdefault(img_id, []).append(ann)

    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        full_image_path = str(Path(subfolder_path) / file_name)
        img_w = float(img_info.get("width", 0))
        img_h = float(img_info.get("height", 0))

        boxes: List[BoundingBox] = []
        if img_w > 0 and img_h > 0:
            for ann in annotations_by_img.get(img_id, []):
                bbox = ann.get("bbox", [])
                if len(bbox) == 4:
                    xmin, ymin, w, h = [float(val) for val in bbox]
                    if w <= 0 or h <= 0:
                        continue
                    cat_id = ann.get("category_id", 0)
                    cat_name = categories.get(cat_id, "grape_bunch")
                    if cat_name not in class_map:
                        continue
                    cls_id = class_map[cat_name]

                    x_center = (xmin + w / 2.0) / img_w
                    y_center = (ymin + h / 2.0) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h

                    box = BoundingBox(class_id=cls_id, x_center=x_center, y_center=y_center,
                                      width=norm_w, height=norm_h)
                    is_valid, _ = validate_bounding_box(box, allowed_classes)
                    if is_valid:
                        boxes.append(box)

        records[file_name] = AnnotationRecord(
            image_path=full_image_path,
            annotation_path=json_path,
            source_format="coco_json",
            boxes=boxes,
            valid=True,
        )

    return records
