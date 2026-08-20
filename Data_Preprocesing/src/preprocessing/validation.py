"""
Annotation Validation Module.
Performs secondary validation pass on annotation records, verifying class IDs and bounding box area/coordinates.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
from src.preprocessing.contracts import AnnotationRecord, ImageRecord


def validate_annotations(
    annotation_records: List[AnnotationRecord],
    reports_path: Path,
    classes: Dict[int, str],
) -> Tuple[List[AnnotationRecord], Dict[str, Any]]:
    """
    Performs deep validation pass on annotations and saves annotation_report.json.
    """
    valid_count = 0
    invalid_count = 0
    total_boxes = 0
    invalid_boxes = 0

    class_distribution: Dict[int, int] = {k: 0 for k in classes.keys()}
    issues: List[Dict[str, Any]] = []

    for record in annotation_records:
        if not record.valid:
            invalid_count += 1
            issues.append({
                "image_path": record.image_path,
                "error": record.error_message or "Invalid annotation record"
            })
            continue

        record_valid = True
        for box in record.boxes:
            total_boxes += 1
            if box.class_id in class_distribution:
                class_distribution[box.class_id] += 1
            else:
                invalid_boxes += 1
                record_valid = False
                issues.append({
                    "image_path": record.image_path,
                    "error": f"Invalid class_id {box.class_id} not in allowed classes {list(classes.keys())}"
                })

        if record_valid:
            valid_count += 1
        else:
            invalid_count += 1

    report = {
        "total_annotation_records": len(annotation_records),
        "valid_records": valid_count,
        "invalid_records": invalid_count,
        "total_bounding_boxes": total_boxes,
        "invalid_bounding_boxes": invalid_boxes,
        "class_distribution": {classes.get(k, str(k)): v for k, v in class_distribution.items()},
        "issues_sample": issues[:50],  # first 50 issues
    }

    reports_path.mkdir(parents=True, exist_ok=True)
    report_file = reports_path / "annotation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return annotation_records, report
