"""
Bounding Box Visualization Generator Module.
Renders bounding boxes on sample images from each dataset subfolder to visually verify coordinate transformation accuracy.
"""

import cv2
from pathlib import Path
from typing import List, Dict
from src.preprocessing.contracts import ImageRecord, AnnotationRecord, BoundingBox


def draw_bounding_boxes(
    image_path: str,
    boxes: List[BoundingBox],
    output_path: str,
    class_map: Dict[int, str],
):
    """Draws green bounding boxes and class labels on an image and saves to output_path."""
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w = img.shape[:2]
    for box in boxes:
        xc = box.x_center * w
        yc = box.y_center * h
        bw = box.width * w
        bh = box.height * h

        xmin = int(round(xc - bw / 2.0))
        ymin = int(round(yc - bh / 2.0))
        xmax = int(round(xc + bw / 2.0))
        ymax = int(round(yc + bh / 2.0))

        cls_name = class_map.get(box.class_id, str(box.class_id))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            img,
            cls_name,
            (xmin, max(ymin - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)


def generate_visualization_samples(
    image_records: List[ImageRecord],
    annotation_records: List[AnnotationRecord],
    output_dir: Path,
    classes: Dict[int, str],
    samples_per_subfolder: int = 3,
):
    """Generates visualization sample images for quality inspection."""
    ann_map = {a.image_path: a for a in annotation_records}
    subfolder_samples: Dict[str, int] = {}

    viz_dir = output_dir / "visualization_samples"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for rec in image_records:
        if rec.status != "valid":
            continue
        sf = rec.subfolder
        count = subfolder_samples.get(sf, 0)
        if count >= samples_per_subfolder:
            continue

        ann = ann_map.get(rec.path)
        if ann and ann.boxes:
            out_path = str(viz_dir / f"{sf}_{rec.filename}")
            draw_bounding_boxes(rec.path, ann.boxes, out_path, classes)
            subfolder_samples[sf] = count + 1
