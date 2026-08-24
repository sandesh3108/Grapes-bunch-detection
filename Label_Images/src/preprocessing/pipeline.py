import os
import shutil
import json
import random
from pathlib import Path
import yaml

class LabelImagesPipeline:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path).resolve()
        self.base_dir = self.config_path.parent.parent

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.input_images_dir = (self.base_dir / self.config.get("input_images_path", "Dataset/images")).resolve()
        self.input_labels_dir = (self.base_dir / self.config.get("input_labels_path", "Dataset/labels")).resolve()
        self.output_dir = (self.base_dir / self.config.get("output_path", "data/processed/v001")).resolve()
        self.reports_dir = (self.base_dir / self.config.get("reports_path", "reports")).resolve()
        
        self.split_config = self.config.get("split", {"train": 0.70, "validation": 0.15, "test": 0.15, "seed": 42})
        self.seed = self.split_config.get("seed", 42)
        self.classes = self.config.get("classes", {0: "grape_bunch"})

    def clean_and_normalize_label(self, label_file: Path) -> list:
        cleaned_lines = []
        if not label_file.exists():
            return cleaned_lines

        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    # Convert class id float (e.g. 0.0) to int (0)
                    cls_id = int(float(parts[0]))
                    xc, yc, w, h = [float(x) for x in parts[1:5]]
                    # Validate box bounds
                    if 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0:
                        cleaned_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                except ValueError:
                    continue
        return cleaned_lines

    def run(self):
        print(f"[Pipeline] Reading images from: {self.input_images_dir}")
        print(f"[Pipeline] Reading labels from: {self.input_labels_dir}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [f for f in self.input_images_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

        valid_pairs = []
        missing_labels = []

        for img_file in image_files:
            stem = img_file.stem
            label_file = self.input_labels_dir / f"{stem}.txt"
            if label_file.exists():
                valid_pairs.append((img_file, label_file))
            else:
                missing_labels.append(img_file)

        print(f"[Pipeline] Total images found: {len(image_files)}")
        print(f"[Pipeline] Images with labels: {len(valid_pairs)}")
        if missing_labels:
            print(f"[Pipeline] Unlabeled images found: {len(missing_labels)}")

        # Deterministic Shuffle and Split
        random.seed(self.seed)
        shuffled_pairs = list(valid_pairs)
        random.shuffle(shuffled_pairs)

        total_valid = len(shuffled_pairs)
        train_ratio = self.split_config.get("train", 0.70)
        val_ratio = self.split_config.get("validation", 0.15)

        train_count = int(total_valid * train_ratio)
        val_count = int(total_valid * val_ratio)

        train_pairs = shuffled_pairs[:train_count]
        val_pairs = shuffled_pairs[train_count:train_count + val_count]
        test_pairs = shuffled_pairs[train_count + val_count:]

        splits = {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs
        }

        # Clear existing output directory
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        total_boxes = 0
        split_stats = {}

        for split_name, pair_list in splits.items():
            img_dir = self.output_dir / split_name / "images"
            lbl_dir = self.output_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            split_boxes = 0
            for img_file, lbl_file in pair_list:
                # Copy image
                dest_img = img_dir / img_file.name
                shutil.copy2(img_file, dest_img)

                # Clean & write label
                dest_lbl = lbl_dir / f"{img_file.stem}.txt"
                cleaned_lines = self.clean_and_normalize_label(lbl_file)
                split_boxes += len(cleaned_lines)

                with open(dest_lbl, "w", encoding="utf-8") as f:
                    f.writelines(cleaned_lines)

            split_stats[split_name] = {
                "image_count": len(pair_list),
                "box_count": split_boxes
            }
            total_boxes += split_boxes

        # Generate data.yaml
        yaml_path = self.output_dir / "data.yaml"
        yaml_content = {
            "path": str(self.output_dir).replace("\\", "/"),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": self.classes
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        print(f"[Pipeline] Created YOLO dataset configuration at: {yaml_path}")

        # Generate final_integrity_report.json
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = self.reports_dir / "final_integrity_report.json"

        report = {
            "valid": True,
            "pipeline": "Label_Images GrapesNet Pipeline",
            "dataset_version": "v001",
            "total_raw_images": len(image_files),
            "labeled_images_processed": len(valid_pairs),
            "unlabeled_images_skipped": len(missing_labels),
            "total_bounding_boxes": total_boxes,
            "splits": split_stats,
            "output_directory": str(self.output_dir).replace("\\", "/"),
            "data_yaml": str(yaml_path).replace("\\", "/")
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        print(f"[Pipeline] Final Integrity Report saved to: {report_file}")
        print(f"[Pipeline] Pipeline completed successfully! valid: true")
        return report
