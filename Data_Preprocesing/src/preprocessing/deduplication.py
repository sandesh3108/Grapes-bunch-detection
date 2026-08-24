"""
Stage 3 — Deduplication Module.
Detects exact duplicates (MD5 hash) and near-duplicates (perceptual hashing: pHash).
Groups duplicates to prevent data leakage across splits.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set, Any, Optional
from PIL import Image
import imagehash
from tqdm import tqdm

from src.preprocessing.contracts import ImageRecord, DuplicateGroup


def compute_file_md5(file_path: str) -> str:
    """Computes MD5 hash of a file's raw bytes."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_perceptual_hash(file_path: str, method: str = "phash") -> Optional[imagehash.ImageHash]:
    """Computes perceptual hash of an image using imagehash."""
    try:
        with Image.open(file_path) as img:
            if method == "dhash":
                return imagehash.dhash(img)
            elif method == "ahash":
                return imagehash.average_hash(img)
            else:
                return imagehash.phash(img)
    except Exception:
        return None


class Deduplicator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("deduplication", {})
        self.method = self.config.get("method", "phash")
        self.threshold = int(self.config.get("threshold", 8))
        self.reports_path = Path(config.get("reports_path", "reports/"))
        self.rejected_path = Path(config.get("rejected_path", "data/rejected/"))

    def process(
        self, image_records: List[ImageRecord]
    ) -> Tuple[List[ImageRecord], List[DuplicateGroup], Dict[str, Any]]:
        """
        Runs exact and near-duplicate detection on valid image records.
        Returns (retained_records, duplicate_groups, report_dict).
        """
        print("--> [Stage 3] Running deduplication...")
        valid_records = [r for r in image_records if not r.corrupted and r.status == "valid"]

        md5_map: Dict[str, List[ImageRecord]] = {}
        phash_map: Dict[str, Tuple[ImageRecord, imagehash.ImageHash]] = {}

        exact_duplicate_paths: Set[str] = set()
        near_duplicate_paths: Set[str] = set()
        duplicate_groups: List[DuplicateGroup] = []

        subfolder_dupes: Dict[str, int] = {}

        # 1. Exact Duplicates via MD5
        print("    Computing MD5 hashes for exact duplicate detection...")
        for rec in tqdm(valid_records, desc="MD5 Check", leave=False):
            md5_val = compute_file_md5(rec.path)
            md5_map.setdefault(md5_val, []).append(rec)

        group_counter = 1
        for md5_val, records in md5_map.items():
            if len(records) > 1:
                rep = records[0]
                dupes = [r.path for r in records[1:]]
                for d in dupes:
                    exact_duplicate_paths.add(d)
                    subfolder_dupes[rep.subfolder] = subfolder_dupes.get(rep.subfolder, 0) + 1

                grp = DuplicateGroup(
                    group_id=f"exact_group_{group_counter}",
                    representative_path=rep.path,
                    duplicate_paths=dupes,
                    subfolder=rep.subfolder,
                    similarity_scores={d: 0.0 for d in dupes},  # 0 Hamming distance
                )
                duplicate_groups.append(grp)
                group_counter += 1

        # 2. Near Duplicates via Perceptual Hash
        non_exact_records = [r for r in valid_records if r.path not in exact_duplicate_paths]
        print(f"    Computing {self.method} perceptual hashes for {len(non_exact_records)} images...")

        hashed_records: List[Tuple[ImageRecord, imagehash.ImageHash]] = []
        for rec in tqdm(non_exact_records, desc="pHash Computation", leave=False):
            ph = compute_perceptual_hash(rec.path, method=self.method)
            if ph is not None:
                hashed_records.append((rec, ph))

        near_group_counter = 1
        visited: Set[str] = set()

        print("    Comparing perceptual hashes...")
        for i in range(len(hashed_records)):
            rec_i, ph_i = hashed_records[i]
            if rec_i.path in visited:
                continue

            current_dupes: List[str] = []
            scores: Dict[str, float] = {}

            for j in range(i + 1, len(hashed_records)):
                rec_j, ph_j = hashed_records[j]
                if rec_j.path in visited:
                    continue

                dist = ph_i - ph_j
                if dist <= self.threshold:
                    current_dupes.append(rec_j.path)
                    scores[rec_j.path] = float(dist)
                    visited.add(rec_j.path)
                    near_duplicate_paths.add(rec_j.path)
                    subfolder_dupes[rec_i.subfolder] = subfolder_dupes.get(rec_i.subfolder, 0) + 1

            if current_dupes:
                grp = DuplicateGroup(
                    group_id=f"near_group_{near_group_counter}",
                    representative_path=rec_i.path,
                    duplicate_paths=current_dupes,
                    subfolder=rec_i.subfolder,
                    similarity_scores=scores,
                )
                duplicate_groups.append(grp)
                near_group_counter += 1

        all_duplicate_paths = exact_duplicate_paths.union(near_duplicate_paths)

        # Update image record statuses for duplicates
        for rec in image_records:
            if rec.path in all_duplicate_paths:
                rec.status = "duplicate"
                rec.rejection_reason = "Duplicate image detected"

        retained_records = [r for r in image_records if r.path not in all_duplicate_paths]

        report = {
            "total_images_evaluated": len(valid_records),
            "exact_duplicates": len(exact_duplicate_paths),
            "near_duplicates": len(near_duplicate_paths),
            "total_duplicates_removed": len(all_duplicate_paths),
            "images_retained": len(retained_records),
            "duplicate_groups_count": len(duplicate_groups),
            "similarity_threshold_used": self.threshold,
            "perceptual_hash_method": self.method,
            "duplicates_per_subfolder": subfolder_dupes,
        }

        self.reports_path.mkdir(parents=True, exist_ok=True)
        report_file = self.reports_path / "deduplication_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"--> [Stage 3 Complete] Found {len(exact_duplicate_paths)} exact and {len(near_duplicate_paths)} near duplicates. Report saved to '{report_file}'.")
        return retained_records, duplicate_groups, report
