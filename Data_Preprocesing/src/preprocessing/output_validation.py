"""Validation of the final YOLO directory, performed before a run is accepted."""

from pathlib import Path
from typing import Any, Dict

from src.preprocessing.annotation.validator import validate_bounding_box
from src.preprocessing.contracts import BoundingBox


def validate_processed_dataset(output_path: Path, classes: Dict[int, str]) -> Dict[str, Any]:
    """Check split structure, image/label pairing, and every emitted YOLO row."""
    errors = []
    splits: Dict[str, Dict[str, int]] = {}
    allowed_image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split in ("train", "val", "test"):
        images_dir = output_path / split / "images"
        labels_dir = output_path / split / "labels"
        images = {path.stem: path for path in images_dir.glob("*") if path.suffix.lower() in allowed_image_extensions}
        labels = {path.stem: path for path in labels_dir.glob("*.txt")}
        splits[split] = {"images": len(images), "labels": len(labels), "boxes": 0}
        if set(images) != set(labels):
            errors.append(f"{split}: image/label stems do not match")

        for stem, label_path in labels.items():
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label_path}:{line_number}: expected 5 fields")
                    continue
                try:
                    raw_class_id = float(parts[0])
                    if not raw_class_id.is_integer():
                        raise ValueError("class id must be an integer")
                    box = BoundingBox(int(raw_class_id), *(float(value) for value in parts[1:]))
                except ValueError:
                    errors.append(f"{label_path}:{line_number}: non-numeric YOLO value")
                    continue
                valid, reason = validate_bounding_box(box, classes)
                if not valid:
                    errors.append(f"{label_path}:{line_number}: {reason}")
                else:
                    splits[split]["boxes"] += 1

    return {"valid": not errors, "splits": splits, "errors": errors[:100]}
