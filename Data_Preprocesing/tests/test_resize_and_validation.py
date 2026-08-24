from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.annotation.validator import sanitize_bounding_box, validate_bounding_box
from src.preprocessing.contracts import BoundingBox
from src.preprocessing.output_validation import validate_processed_dataset
from src.preprocessing.resize import letterbox_image, transform_bbox_letterbox


CLASSES = {0: "grape_bunch"}


def test_letterbox_bbox_stays_valid():
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    _, scale, pad_w, pad_h = letterbox_image(image, target_size=640)
    transformed = transform_bbox_letterbox(
        BoundingBox(0, 0.5, 0.5, 0.5, 0.5), 400, 200, 640, scale, pad_w, pad_h
    )
    assert validate_bounding_box(transformed, CLASSES)[0]
    assert transformed.y_center == 0.5


def test_corner_clipping_preserves_valid_geometry():
    clipped = sanitize_bounding_box(BoundingBox(0, 0.05, 0.5, 0.4, 0.4))
    assert clipped.x_center == 0.125
    assert clipped.width == 0.25
    assert validate_bounding_box(clipped, CLASSES)[0]


def test_final_validator_rejects_bad_yolo_label(tmp_path: Path):
    for split in ("train", "val", "test"):
        images = tmp_path / split / "images"
        labels = tmp_path / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        cv2.imwrite(str(images / "sample.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
        (labels / "sample.txt").write_text("0 1.1 0.5 0.2 0.2\n", encoding="utf-8")
    report = validate_processed_dataset(tmp_path, CLASSES)
    assert not report["valid"]
