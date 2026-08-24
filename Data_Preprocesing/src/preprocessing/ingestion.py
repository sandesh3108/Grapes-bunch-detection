"""
Stage 1 — Ingestion and Format Detection Module.
Recursively scans input dataset, auto-detects per-subfolder annotation formats,
checks image integrity, matches annotations, and generates ingestion_report.json.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
from tqdm import tqdm

from src.preprocessing.contracts import ImageRecord, AnnotationRecord
from src.preprocessing.annotation.format_detector import detect_annotation_format, find_all_images, IMAGE_EXTENSIONS
from src.preprocessing.annotation.voc_to_yolo import parse_voc_xml
from src.preprocessing.annotation.coco_to_yolo import parse_coco_json
from src.preprocessing.annotation.yolo_passthrough_validator import parse_yolo_txt
from src.preprocessing.annotation.custom_to_yolo import handle_custom_format


GENERATED_VARIANT_PREFIXES = ("horflip_", "vertflip_", "topwarp_", "bottomwarp_", "leftwarp_", "rightomwarp_")


def classify_modality(image_path: Path) -> str:
    """Classify GrapesNet-style RGB-D files without modifying the source data."""
    parts = {part.lower() for part in image_path.parts}
    name = image_path.name.lower()
    if "rgb-d" in parts:
        return "rgb" if "_color" in name else "depth"
    return "rgb"


def provenance_group_for(image_path: Path) -> str:
    """Keep published flip/warp derivatives with their unmodified source image."""
    stem = image_path.stem
    lower = stem.lower()
    for prefix in GENERATED_VARIANT_PREFIXES:
        if lower.startswith(prefix):
            return stem[len(prefix):]
    return stem


def validate_image_file(image_path: str) -> Tuple[bool, int, int, int, str, Optional[str]]:
    """
    Validates image readability using Pillow.
    Returns (is_valid, width, height, channels, format, error_msg).
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Re-open after verify() to get properties safely
        with Image.open(image_path) as img:
            w, h = img.size
            fmt = (img.format or "").lower()
            mode = img.mode
            channels = len(mode) if mode else 3
            if mode == "RGB":
                channels = 3
            elif mode == "RGBA":
                channels = 4
            elif mode == "L":
                channels = 1

            if w <= 0 or h <= 0:
                return False, 0, 0, 0, fmt, "Invalid dimensions (<= 0)"
            return True, w, h, channels, fmt, None
    except Exception as e:
        return False, 0, 0, 0, "", f"Corrupted image file: {str(e)}"


class DatasetIngestor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_path = Path(config.get("input_path", "Dataset/GrapesNet"))
        self.rejected_path = Path(config.get("rejected_path", "data/rejected/"))
        self.reports_path = Path(config.get("reports_path", "reports/"))
        self.classes = {int(k): v for k, v in config.get("classes", {0: "grape_bunch"}).items()}
        self.class_name_map = {v: k for k, v in self.classes.items()}
        input_config = config.get("input", {})
        self.modalities = set(input_config.get("modalities", ["rgb"]))
        self.exclude_generated_variants = bool(input_config.get("exclude_generated_variants", True))
        labels_dir = input_config.get("yolo_labels_dir")
        self.yolo_labels_dir = Path(labels_dir) if labels_dir else None
        self.annotation_mode = config.get("annotation", {}).get("mode", "require")
        if self.annotation_mode not in {"require", "manual-later"}:
            raise ValueError("annotation.mode must be 'require' or 'manual-later'")

    def run(self) -> Tuple[List[ImageRecord], List[AnnotationRecord], Dict[str, Any]]:
        """
        Executes dataset ingestion.
        Returns (image_records, annotation_records, report_dict).
        """
        # Robust path resolution
        if not self.input_path.exists():
            parent_rel = Path("..") / self.input_path
            if parent_rel.exists():
                self.input_path = parent_rel
            else:
                cwd_rel = Path.cwd() / self.input_path
                if cwd_rel.exists():
                    self.input_path = cwd_rel

        print(f"--> [Stage 1] Scanning dataset at '{self.input_path.resolve()}'...")
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")

        if self.yolo_labels_dir and not self.yolo_labels_dir.is_absolute():
            self.yolo_labels_dir = (Path.cwd() / self.yolo_labels_dir).resolve()
        if self.yolo_labels_dir and not self.yolo_labels_dir.is_dir():
            raise FileNotFoundError(f"YOLO labels directory does not exist: {self.yolo_labels_dir}")

        # Discover top-level subfolders or process as a single dataset if no subfolders exist
        subfolders = [d for d in self.input_path.iterdir() if d.is_dir()]
        if not subfolders:
            subfolders = [self.input_path]

        all_image_records: List[ImageRecord] = []
        all_annotation_records: List[AnnotationRecord] = []

        report_summary: Dict[str, Any] = {
            "total_images": 0,
            "valid_images": 0,
            "corrupted_images": 0,
            "unannotated_images": 0,
            "excluded_non_rgb_images": 0,
            "excluded_generated_variants": 0,
            "orphan_annotations": 0,
            "detected_annotation_format_per_subfolder": {},
            "subfolder_breakdown": {},
        }

        for subfolder in subfolders:
            subfolder_name = subfolder.name
            fmt = "yolo_txt" if self.yolo_labels_dir else detect_annotation_format(str(subfolder))
            report_summary["detected_annotation_format_per_subfolder"][subfolder_name] = fmt
            print(f"    Subfolder '{subfolder_name}': Detected format = '{fmt}'")

            discovered_images = find_all_images(str(subfolder))
            images = []
            for candidate in discovered_images:
                modality = classify_modality(candidate)
                if modality not in self.modalities:
                    report_summary["excluded_non_rgb_images"] += 1
                    continue
                if self.exclude_generated_variants and candidate.stem.lower().startswith(GENERATED_VARIANT_PREFIXES):
                    report_summary["excluded_generated_variants"] += 1
                    continue
                images.append(candidate)
            subfolder_report = {
                "total_images": len(images),
                "valid": 0,
                "corrupted": 0,
                "unannotated": 0,
                "format": fmt,
            }

            # If COCO format, pre-parse the COCO json file
            coco_records: Dict[str, AnnotationRecord] = {}
            if fmt == "coco_json":
                json_files = list(subfolder.rglob("*.json"))
                if json_files:
                    coco_records = parse_coco_json(
                        str(json_files[0]),
                        str(subfolder),
                        self.class_name_map,
                        self.classes,
                    )

            for img_path in tqdm(images, desc=f"Ingesting {subfolder_name}", leave=False):
                rel_path = str(img_path)
                filename = img_path.name
                stem = img_path.stem
                modality = classify_modality(img_path)

                is_valid_img, w, h, channels, img_fmt, err_msg = validate_image_file(str(img_path))
                if not is_valid_img:
                    rec = ImageRecord(
                        path=rel_path,
                        subfolder=subfolder_name,
                        filename=filename,
                        corrupted=True,
                        status="corrupted",
                        rejection_reason=err_msg,
                        detected_annotation_format=fmt,
                    )
                    all_image_records.append(rec)
                    subfolder_report["corrupted"] += 1
                    continue

                # Locate corresponding annotation file
                ann_rec: Optional[AnnotationRecord] = None
                ann_path: Optional[str] = None

                if fmt == "voc_xml":
                    possible_xml = img_path.with_suffix(".xml")
                    if possible_xml.exists():
                        ann_path = str(possible_xml)
                        ann_rec = parse_voc_xml(
                            ann_path, rel_path, w, h, self.class_name_map, self.classes
                        )
                elif fmt == "yolo_txt":
                    possible_txt = (self.yolo_labels_dir / f"{img_path.stem}.txt") if self.yolo_labels_dir else img_path.with_suffix(".txt")
                    if possible_txt.exists():
                        ann_path = str(possible_txt)
                        ann_rec = parse_yolo_txt(ann_path, rel_path, self.classes)
                elif fmt == "coco_json":
                    if filename in coco_records:
                        ann_rec = coco_records[filename]
                        ann_path = ann_rec.annotation_path
                    elif stem in coco_records:
                        ann_rec = coco_records[stem]
                        ann_path = ann_rec.annotation_path

                status = "valid"
                rejection_reason = None
                if fmt == "none" or ann_rec is None:
                    subfolder_report["unannotated"] += 1
                    status = "unannotated"
                    rejection_reason = "No supported human-authored annotation found"
                else:
                    if not ann_rec.valid:
                        status = "invalid_annotation"
                        rejection_reason = ann_rec.error_message or "Invalid annotation"
                    elif not ann_rec.boxes:
                        status = "invalid_annotation"
                        rejection_reason = "Annotation contains no valid bounding boxes"
                    else:
                        subfolder_report["valid"] += 1

                img_record = ImageRecord(
                    path=rel_path,
                    subfolder=subfolder_name,
                    filename=filename,
                    width=w,
                    height=h,
                    channels=channels,
                    format=img_fmt,
                    corrupted=False,
                    annotation_path=ann_path,
                    detected_annotation_format=fmt,
                    modality=modality,
                    provenance_group=provenance_group_for(img_path),
                    status=status,
                    rejection_reason=rejection_reason,
                )

                all_image_records.append(img_record)
                if ann_rec is not None:
                    all_annotation_records.append(ann_rec)

            report_summary["subfolder_breakdown"][subfolder_name] = subfolder_report
            report_summary["total_images"] += subfolder_report["total_images"]
            report_summary["valid_images"] += subfolder_report["valid"]
            report_summary["corrupted_images"] += subfolder_report["corrupted"]
            report_summary["unannotated_images"] += subfolder_report["unannotated"]

        # Save report
        self.reports_path.mkdir(parents=True, exist_ok=True)
        report_file = self.reports_path / "ingestion_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_summary, f, indent=2)

        print(f"--> [Stage 1 Complete] Found {report_summary['total_images']} total images. Report saved to '{report_file}'.")
        return all_image_records, all_annotation_records, report_summary
