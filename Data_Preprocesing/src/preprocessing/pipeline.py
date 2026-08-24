"""
Pipeline Orchestrator Module.
Connects all 8 stages of the data engineering pipeline in order, writing processed output to data/processed/
and reports to reports/ and experiments/dataset_v001/.
"""

import shutil
import json
import random
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from src.preprocessing.ingestion import DatasetIngestor
from src.preprocessing.validation import validate_annotations
from src.preprocessing.deduplication import Deduplicator
from src.preprocessing.quality_filter import QualityFilter
from src.preprocessing.splitting import GroupAwareSplitter
from src.preprocessing.resize import letterbox_image, transform_bbox_letterbox
from src.preprocessing.normalization import get_normalization_policy
from src.preprocessing.augmentation import augment_training_sample
from src.preprocessing.statistics import generate_dataset_statistics
from src.preprocessing.visualization import generate_visualization_samples
from src.preprocessing.contracts import AnnotationRecord, ImageRecord, BoundingBox
from src.preprocessing.output_validation import validate_processed_dataset
from src.preprocessing.configuration import validate_and_normalize_config


class PreprocessingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = validate_and_normalize_config(config)
        config = self.config
        self.output_path = Path(config.get("output_path", "data/processed/"))
        self.rejected_path = Path(config.get("rejected_path", "data/rejected/"))
        self.reports_path = Path(config.get("reports_path", "reports/"))
        self.experiments_path = Path(config.get("experiments_path", "experiments/dataset_v001/"))
        self.classes = {int(k): v for k, v in config.get("classes", {0: "grape_bunch"}).items()}
        self.target_size = int(config.get("resize", {}).get("target_size", 640))
        self.padding_color = tuple(config.get("resize", {}).get("padding_color", [114, 114, 114]))
        self.annotation_mode = config.get("annotation", {}).get("mode", "require")
        self.overwrite_output = bool(config.get("output", {}).get("overwrite", False))

    def run(self, validate_only: bool = False, stage: Optional[str] = None):
        """Executes the complete pipeline or specified stage."""
        print("\n========================================================")
        print("  Starting GrapesNet Preprocessing Pipeline Execution")
        print("========================================================\n")

        # Snapshot config into experiment folder
        self.experiments_path.mkdir(parents=True, exist_ok=True)
        with open(self.experiments_path / "preprocessing_config_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        seed = int(self.config.get("reproducibility", {}).get("seed", self.config.get("split", {}).get("seed", 42)))
        random.seed(seed)
        np.random.seed(seed)
        cv2.setRNGSeed(seed)

        # Stage 1: Ingestion & Format Auto-detection
        ingestor = DatasetIngestor(self.config)
        image_records, annotation_records, ingestion_report = ingestor.run()

        # Secondary Validation Pass
        annotation_records, annotation_report = validate_annotations(
            annotation_records, self.reports_path, self.classes
        )
        image_by_path = {record.path: record for record in image_records}
        for annotation in annotation_records:
            if not annotation.valid:
                image = image_by_path.get(annotation.image_path)
                if image:
                    image.status = "invalid_annotation"
                    image.rejection_reason = annotation.error_message or "Invalid annotation"

        if validate_only or stage == "ingestion" or stage == "validation":
            print("--> Stopped after Validation phase (validate-only mode).")
            return

        annotated_images = [record for record in image_records if record.status == "valid"]
        if not annotated_images:
            message = (
                "No valid human-authored annotations were found. No YOLO training dataset was created. "
                "Annotate the RGB images and rerun the pipeline."
            )
            if self.annotation_mode == "manual-later":
                print(f"--> {message} Manual-later mode stops after validation by design.")
                return
            raise RuntimeError(message)

        # Stage 2: Annotation Standardization (implicit in ingestion conversion to normalized BoundingBox contracts)

        # Stage 3: Deduplication
        deduplicator = Deduplicator(self.config)
        retained_images, duplicate_groups, dedup_report = deduplicator.process(image_records)

        if stage == "deduplication":
            print("--> Stopped after Stage 3 (Deduplication).")
            return

        # Stage 4: Quality Filtering
        quality_filter = QualityFilter(self.config)
        clean_images, clean_annotations, quality_report = quality_filter.process(
            retained_images, annotation_records
        )

        if stage == "quality_filter":
            print("--> Stopped after Stage 4 (Quality Filtering).")
            return

        # Stage 5: Group-Aware Dataset Splitting
        splitter = GroupAwareSplitter(self.config)
        split_assignments, split_report = splitter.process(
            clean_images, clean_annotations, duplicate_groups
        )

        if stage == "split":
            print("--> Stopped after Stage 5 (Dataset Splitting).")
            return

        # Stage 6 & Stage 8: Resize/Letterbox + Train Augmentation & Output Generation
        print("--> [Stage 6-8] Processing images, resizing/letterboxing, augmenting train split, and saving dataset...")

        ann_map = {a.image_path: a for a in clean_annotations}
        split_map = {s.image_path: s.split for s in split_assignments}

        # Never discard a previous dataset unless the configuration explicitly permits it.
        if self.output_path.exists():
            if not self.overwrite_output:
                raise FileExistsError(
                    f"Output path already exists: {self.output_path}. Set output.overwrite=true only after reviewing it."
                )
            shutil.rmtree(self.output_path)

        for split_name in ["train", "val", "test"]:
            (self.output_path / split_name / "images").mkdir(parents=True, exist_ok=True)
            (self.output_path / split_name / "labels").mkdir(parents=True, exist_ok=True)

        for img_rec in tqdm(clean_images, desc="Processing Output Images", leave=False):
            split_name = split_map.get(img_rec.path)
            if not split_name:
                continue

            # Read original image
            img_bgr = cv2.imread(img_rec.path)
            if img_bgr is None:
                raise RuntimeError(f"OpenCV could not read validated image: {img_rec.path}")

            orig_h, orig_w = img_bgr.shape[:2]

            # Stage 6: Letterbox resize to 640x640
            letterboxed_img, scale, pad_w, pad_h = letterbox_image(
                img_bgr, target_size=self.target_size, padding_color=self.padding_color
            )

            ann_rec = ann_map.get(img_rec.path)
            transformed_boxes: List[BoundingBox] = []
            if ann_rec and ann_rec.boxes:
                for b in ann_rec.boxes:
                    tb = transform_bbox_letterbox(
                        b, orig_w, orig_h, self.target_size, scale, pad_w, pad_h
                    )
                    transformed_boxes.append(tb)

            # Stage 8: Apply Train-only Augmentation
            if split_name == "train":
                letterboxed_img, transformed_boxes = augment_training_sample(
                    letterboxed_img, transformed_boxes, self.config
                )

            # Save processed image & label file
            dest_img_path = self.output_path / split_name / "images" / f"{img_rec.subfolder}_{img_rec.filename}"
            if not cv2.imwrite(str(dest_img_path), letterboxed_img):
                raise RuntimeError(f"Failed to write processed image: {dest_img_path}")

            dest_label_path = (
                self.output_path / split_name / "labels" / f"{img_rec.subfolder}_{Path(img_rec.filename).stem}.txt"
            )
            with open(dest_label_path, "w", encoding="utf-8") as f:
                for b in transformed_boxes:
                    f.write(b.to_yolo_line() + "\n")

        # Quarantined / Rejected Files Handling
        self._quarantine_files(image_records)

        # Stage 7: Normalization Policy Record
        norm_policy = get_normalization_policy(self.config)

        # Generate Statistics and Visualizations
        print("--> Generating dataset statistics and visualization samples...")
        generate_dataset_statistics(
            clean_images, clean_annotations, split_assignments, self.experiments_path
        )
        generate_visualization_samples(
            clean_images, clean_annotations, self.experiments_path, self.classes
        )
        integrity_report = validate_processed_dataset(self.output_path, self.classes)
        with open(self.reports_path / "final_integrity_report.json", "w", encoding="utf-8") as f:
            json.dump(integrity_report, f, indent=2)
        if not integrity_report["valid"]:
            raise RuntimeError("Final output validation failed; see final_integrity_report.json")

        print("\n========================================================")
        print(f"  Pipeline Execution Successfully Completed!")
        print(f"  Processed dataset saved to: {self.output_path}")
        print(f"  Reports saved to:           {self.reports_path}")
        print(f"  Experiment data saved to:   {self.experiments_path}")
        print("========================================================\n")

    def _quarantine_files(self, image_records: List[ImageRecord]):
        """Copy quarantined/rejected files to appropriate data/rejected/ folders."""
        for rec in image_records:
            if rec.status not in ("valid", "unannotated"):
                target_dir = self.rejected_path / rec.status
                target_dir.mkdir(parents=True, exist_ok=True)
                dest_file = target_dir / f"{rec.subfolder}_{rec.filename}"
                if Path(rec.path).exists() and not dest_file.exists():
                    try:
                        shutil.copy2(rec.path, dest_file)
                    except Exception:
                        pass
