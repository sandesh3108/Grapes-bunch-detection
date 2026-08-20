"""
Stage 5 — Group-Aware Dataset Splitting Module.
Performs deterministic, leakage-safe splitting (70% train / 15% val / 15% test).
Grouping is aware of subfolders (Dataset 1..4) and duplicate groups to prevent data leakage.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict

from src.preprocessing.contracts import ImageRecord, AnnotationRecord, DuplicateGroup, SplitAssignment


class GroupAwareSplitter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("split", {})
        self.train_ratio = float(self.config.get("train", 0.70))
        self.val_ratio = float(self.config.get("validation", 0.15))
        self.test_ratio = float(self.config.get("test", 0.15))
        self.seed = int(self.config.get("seed", 42))
        self.reports_path = Path(config.get("reports_path", "reports/"))

    def process(
        self,
        image_records: List[ImageRecord],
        annotation_records: List[AnnotationRecord],
        duplicate_groups: List[DuplicateGroup],
    ) -> Tuple[List[SplitAssignment], Dict[str, Any]]:
        """
        Executes group-aware split across image records.
        Returns (assignments, report_dict).
        """
        print(f"--> [Stage 5] Running Group-Aware Dataset Split (train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}, seed={self.seed})...")
        random.seed(self.seed)

        # Build duplicate group lookup: image_path -> group_id
        dupe_lookup: Dict[str, str] = {}
        for grp in duplicate_groups:
            for path in [grp.representative_path] + grp.duplicate_paths:
                dupe_lookup[path] = grp.group_id

        # Organize images by subfolder and group
        subfolder_groups: Dict[str, Dict[str, List[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
        for rec in image_records:
            if rec.status not in ("valid", "unannotated"):
                continue
            grp_id = dupe_lookup.get(rec.path, f"single_{rec.path}")
            subfolder_groups[rec.subfolder][grp_id].append(rec)

        assignments: List[SplitAssignment] = []
        subfolder_stats: Dict[str, Dict[str, int]] = {}

        for subfolder_name, groups_dict in subfolder_groups.items():
            group_keys = list(groups_dict.keys())
            random.shuffle(group_keys)

            n_groups = len(group_keys)
            n_train = int(round(n_groups * self.train_ratio))
            n_val = int(round(n_groups * self.val_ratio))
            # Ensure rest goes to test
            n_test = n_groups - n_train - n_val
            if n_test < 0:
                n_test = 0

            train_keys = set(group_keys[:n_train])
            val_keys = set(group_keys[n_train:n_train + n_val])
            test_keys = set(group_keys[n_train + n_val:])

            counts = {"train": 0, "val": 0, "test": 0}

            for grp_id, records in groups_dict.items():
                if grp_id in train_keys:
                    split_label = "train"
                elif grp_id in val_keys:
                    split_label = "val"
                else:
                    split_label = "test"

                for rec in records:
                    assignments.append(
                        SplitAssignment(
                            image_path=rec.path,
                            split=split_label,
                            group_id=grp_id,
                            subfolder=rec.subfolder,
                        )
                    )
                    counts[split_label] += 1

            subfolder_stats[subfolder_name] = counts

        # Compute summary stats per split
        ann_map = {a.image_path: a for a in annotation_records}
        split_counts: Dict[str, Dict[str, int]] = {
            "train": {"images": 0, "annotations": 0, "boxes": 0},
            "val": {"images": 0, "annotations": 0, "boxes": 0},
            "test": {"images": 0, "annotations": 0, "boxes": 0},
        }

        for assign in assignments:
            sp = assign.split
            split_counts[sp]["images"] += 1
            ann_rec = ann_map.get(assign.image_path)
            if ann_rec and ann_rec.boxes:
                split_counts[sp]["annotations"] += 1
                split_counts[sp]["boxes"] += len(ann_rec.boxes)

        report = {
            "ratios": {
                "train": self.train_ratio,
                "validation": self.val_ratio,
                "test": self.test_ratio,
            },
            "random_seed": self.seed,
            "overall_counts": split_counts,
            "per_subfolder_breakdown": subfolder_stats,
        }

        self.reports_path.mkdir(parents=True, exist_ok=True)
        report_file = self.reports_path / "split_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"--> [Stage 5 Complete] Split into {split_counts['train']['images']} train, {split_counts['val']['images']} val, {split_counts['test']['images']} test. Report saved to '{report_file}'.")
        return assignments, report
