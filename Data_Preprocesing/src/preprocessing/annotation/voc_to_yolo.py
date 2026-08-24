"""
Pascal VOC XML to YOLO Converter.
Parses Pascal VOC .xml annotation files and converts bounding boxes to normalized YOLO format.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
from src.preprocessing.contracts import AnnotationRecord, BoundingBox
from src.preprocessing.annotation.validator import validate_bounding_box, sanitize_bounding_box


def parse_voc_xml(
    xml_path: str,
    image_path: str,
    image_width: int,
    image_height: int,
    class_map: Dict[str, int],
    allowed_classes: Dict[int, str],
) -> AnnotationRecord:
    """Converts a Pascal VOC XML file to an AnnotationRecord with normalized YOLO BoundingBoxes."""
    path = Path(xml_path)
    if not path.exists():
        return AnnotationRecord(
            image_path=image_path,
            annotation_path=xml_path,
            source_format="voc_xml",
            valid=False,
            error_message=f"VOC XML file not found: {xml_path}",
        )

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        return AnnotationRecord(
            image_path=image_path,
            annotation_path=xml_path,
            source_format="voc_xml",
            valid=False,
            error_message=f"XML parse error: {str(e)}",
        )

    # Use size from XML if image_width / image_height are not provided
    size_elem = root.find("size")
    if size_elem is not None:
        try:
            w_val = int(size_elem.findtext("width", "0"))
            h_val = int(size_elem.findtext("height", "0"))
            if w_val > 0 and h_val > 0:
                image_width = w_val
                image_height = h_val
        except ValueError:
            pass

    if image_width <= 0 or image_height <= 0:
        return AnnotationRecord(
            image_path=image_path,
            annotation_path=xml_path,
            source_format="voc_xml",
            valid=False,
            error_message="Invalid or missing image dimensions in VOC XML.",
        )

    boxes: List[BoundingBox] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in class_map:
            return AnnotationRecord(
                image_path=image_path,
                annotation_path=xml_path,
                source_format="voc_xml",
                valid=False,
                error_message=f"Unknown VOC class '{name}'",
            )
        cls_id = class_map[name]

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
        except ValueError:
            continue

        width = xmax - xmin
        height = ymax - ymin
        if width <= 0 or height <= 0:
            continue

        x_center = (xmin + width / 2.0) / image_width
        y_center = (ymin + height / 2.0) / image_height
        norm_width = width / image_width
        norm_height = height / image_height

        box = BoundingBox(class_id=cls_id, x_center=x_center, y_center=y_center,
                          width=norm_width, height=norm_height)

        is_valid, err_msg = validate_bounding_box(box, allowed_classes)
        if is_valid:
            boxes.append(box)

    return AnnotationRecord(
        image_path=image_path,
        annotation_path=xml_path,
        source_format="voc_xml",
        boxes=boxes,
        valid=True,
    )
