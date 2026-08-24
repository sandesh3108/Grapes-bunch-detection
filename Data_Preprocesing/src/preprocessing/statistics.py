"""
Dataset Statistics Generator Module.
Calculates pre/post processing dataset statistics, bbox area distributions, objects-per-image metrics,
and per-subfolder breakdowns, saving output to experiments/dataset_v001/dataset_statistics.json.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from src.preprocessing.contracts import ImageRecord, AnnotationRecord, SplitAssignment


def generate_dataset_statistics(
    image_records: List[ImageRecord],
    annotation_records: List[AnnotationRecord],
    split_assignments: List[SplitAssignment],
    output_dir: Path,
) -> Dict[str, Any]:
    """Generates comprehensive dataset statistics and saves to output_dir/dataset_statistics.json."""
    ann_map = {a.image_path: a for a in annotation_records}
    split_map = {s.image_path: s.split for s in split_assignments}

    stats: Dict[str, Any] = {
        "overall": {
            "total_images": len(image_records),
            "valid_retained_images": len([r for r in image_records if r.status == "valid"]),
            "quarantined_images": len([r for r in image_records if r.status != "valid"]),
        },
        "by_subfolder": {},
        "by_split": {"train": {}, "val": {}, "test": {}},
        "bounding_box_metrics": {},
    }

    subfolder_stats: Dict[str, Dict[str, Any]] = {}
    for rec in image_records:
        sf = rec.subfolder
        if sf not in subfolder_stats:
            subfolder_stats[sf] = {"total": 0, "valid": 0, "quarantined": 0, "boxes": 0}
        subfolder_stats[sf]["total"] += 1
        if rec.status == "valid":
            subfolder_stats[sf]["valid"] += 1
            ann = ann_map.get(rec.path)
            if ann and ann.boxes:
                subfolder_stats[sf]["boxes"] += len(ann.boxes)
        else:
            subfolder_stats[sf]["quarantined"] += 1

    stats["by_subfolder"] = subfolder_stats

    for assignment in split_assignments:
        split = assignment.split
        entry = stats["by_split"][split]
        entry["images"] = entry.get("images", 0) + 1
        annotation = ann_map.get(assignment.image_path)
        box_count = len(annotation.boxes) if annotation else 0
        entry["annotated_images"] = entry.get("annotated_images", 0) + int(box_count > 0)
        entry["boxes"] = entry.get("boxes", 0) + box_count

    # Bounding box area metrics
    all_areas = []
    all_widths = []
    all_heights = []
    boxes_per_image = []

    for rec in image_records:
        if rec.status != "valid":
            continue
        ann = ann_map.get(rec.path)
        if ann:
            boxes_per_image.append(len(ann.boxes))
            for b in ann.boxes:
                all_areas.append(b.width * b.height)
                all_widths.append(b.width)
                all_heights.append(b.height)

    if all_areas:
        stats["bounding_box_metrics"] = {
            "total_boxes": len(all_areas),
            "mean_area": float(np.mean(all_areas)),
            "std_area": float(np.std(all_areas)),
            "min_area": float(np.min(all_areas)),
            "max_area": float(np.max(all_areas)),
            "mean_width": float(np.mean(all_widths)),
            "mean_height": float(np.mean(all_heights)),
            "mean_objects_per_image": float(np.mean(boxes_per_image)) if boxes_per_image else 0.0,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = output_dir / "dataset_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"--> Dataset statistics saved to '{stats_file}'.")
    return stats
