"""
Stage 4 — Quality Filtering Module.
Evaluates blur using OpenCV Laplacian Variance and filters out problematic/extremely small bounding boxes.
"""

import json
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

from src.preprocessing.contracts import ImageRecord, AnnotationRecord


def compute_blur_variance(image_path: str) -> float:
    """Computes Laplacian variance to measure image sharpness/blur level."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0


class QualityFilter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("quality", {})
        self.blur_config = self.config.get("blur", {})
        self.blur_enabled = self.blur_config.get("enabled", True)
        self.blur_threshold = float(self.blur_config.get("threshold", 100.0))

        self.min_box_area_ratio = float(self.config.get("min_box_area_ratio", 0.0001))
        self.max_box_area_ratio = float(self.config.get("max_box_area_ratio", 0.99))

        self.reports_path = Path(config.get("reports_path", "reports/"))

    def process(
        self,
        image_records: List[ImageRecord],
        annotation_records: List[AnnotationRecord],
    ) -> Tuple[List[ImageRecord], List[AnnotationRecord], Dict[str, Any]]:
        """
        Runs quality filtering on retained image records and annotations.
        Returns (clean_image_records, clean_annotation_records, report_dict).
        """
        print("--> [Stage 4] Running quality filtering...")
        ann_map: Dict[str, AnnotationRecord] = {a.image_path: a for a in annotation_records}

        retained_images: List[ImageRecord] = []
        retained_annotations: List[AnnotationRecord] = []

        blurred_count = 0
        invalid_box_count = 0
        rejection_details: List[Dict[str, Any]] = []

        subfolder_quarantined: Dict[str, int] = {}

        for rec in tqdm(image_records, desc="Quality Check", leave=False):
            if rec.corrupted or rec.status != "valid":
                continue

            # 1. Blur Check
            if self.blur_enabled:
                blur_score = compute_blur_variance(rec.path)
                if blur_score < self.blur_threshold:
                    rec.status = "low_quality"
                    rec.rejection_reason = f"Image blur score {blur_score:.2f} below threshold {self.blur_threshold}"
                    blurred_count += 1
                    subfolder_quarantined[rec.subfolder] = subfolder_quarantined.get(rec.subfolder, 0) + 1
                    rejection_details.append({
                        "image_path": rec.path,
                        "subfolder": rec.subfolder,
                        "reason": rec.rejection_reason,
                        "blur_score": blur_score,
                    })
                    continue

            # 2. Annotation Quality Check
            ann_rec = ann_map.get(rec.path)
            if ann_rec and ann_rec.boxes:
                valid_boxes = []
                for box in ann_rec.boxes:
                    area = box.width * box.height
                    if self.min_box_area_ratio <= area <= self.max_box_area_ratio:
                        valid_boxes.append(box)
                    else:
                        invalid_box_count += 1

                ann_rec.boxes = valid_boxes

            retained_images.append(rec)
            if ann_rec:
                retained_annotations.append(ann_rec)

        report = {
            "total_images_checked": len([r for r in image_records if r.status in ("valid", "low_quality")]),
            "blurred_images_quarantined": blurred_count,
            "invalid_area_boxes_removed": invalid_box_count,
            "images_passed": len(retained_images),
            "blur_threshold": self.blur_threshold,
            "min_box_area_ratio": self.min_box_area_ratio,
            "max_box_area_ratio": self.max_box_area_ratio,
            "quarantined_per_subfolder": subfolder_quarantined,
            "sample_rejections": rejection_details[:50],
        }

        self.reports_path.mkdir(parents=True, exist_ok=True)
        report_file = self.reports_path / "quality_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"--> [Stage 4 Complete] Quarantined {blurred_count} blurred images. Report saved to '{report_file}'.")
        return retained_images, retained_annotations, report
