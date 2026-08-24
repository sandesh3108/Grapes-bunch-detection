# Technical Architecture

This document describes the technical structure of the preprocessing pipeline **now implemented** in `Data Preprocesing/`, following `PREPROCESSING_PIPELINE.md`. It exists so any developer or researcher can understand what each module does and how data flows through the system.

> **Status:** The pipeline is operational but deliberately blocks YOLO dataset generation until reviewed human-authored annotations are available. See `ANNOTATION_AND_PREPROCESSING_GUIDE.md`.

---

## 1. High-Level Architecture

```text
                ┌─────────────────────┐
                │  configs/preprocessing.yaml │
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │  scripts/preprocess.py │  ← CLI entrypoint
                └──────────┬───────────┘
                           │
                ┌──────────▼───────────┐
                │  src/preprocessing/pipeline.py │  ← orchestrator
                └──────────┬───────────┘
                           │
      ┌────────┬───────────┼───────────┬─────────┬──────────┐
      ▼        ▼           ▼           ▼         ▼          ▼
 ingestion  annotation  dedup     quality_filter split   resize/norm/
   .py        /*.py      .py         .py        .py     augmentation
```

Every stage is a standalone module with a defined input/output contract. `pipeline.py` calls them in sequence, and each stage writes both processed data (to `data/`) and a JSON report (to `reports/`).

---

## 2. Module Reference

### `src/preprocessing/ingestion.py`
- **Responsibility:** Recursively scan `data/raw/` (mirroring the user's `Dataset/` folder — see Section 0 of `PREPROCESSING_PIPELINE.md`), discover images and annotation files, verify pairing, and detect corrupted/invalid files.
- **Key functions (expected):**
  - `scan_dataset(root_path) -> List[ImageRecord]`
  - `validate_image(path) -> ValidationResult`
  - `match_images_to_annotations(images, annotations) -> MatchResult`
- **Depends on:** `annotation/format_detector.py` (to determine what kind of annotation file, if any, exists per `Dataset-N` subfolder)
- **Output:** `reports/ingestion_report.json`
- **Likely libraries:** `Pillow` or `opencv-python` (image reading/corruption checks), `pathlib`, `os`

### `src/preprocessing/validation.py`
- **Responsibility:** Deeper validation pass — annotation syntax, class ID validity (must be exactly `0` for `grape_bunch`), bounding-box coordinate sanity (in-bounds, non-zero area).
- **Key functions:** `validate_annotation_file(path, class_map) -> List[ValidationError]`
- **Output:** feeds into `reports/annotation_report.json`

### `src/preprocessing/annotation/`
- **`format_detector.py`** — inspects a subfolder's annotation files and returns one of: `"voc_xml"`, `"coco_json"`, `"yolo_txt"`, `"none"`. Must run independently per `Dataset-N` subfolder, since GrapesNet subsets may not share a format.
- **`voc_to_yolo.py`** — converts Pascal VOC `.xml` → YOLO `.txt`.
- **`coco_to_yolo.py`** — converts a COCO `.json` → per-image YOLO `.txt` files.
- **`yolo_passthrough_validator.py`** — if already YOLO format, validates rather than converts.
- **`custom_to_yolo.py`** — placeholder for non-standard formats with actionable error handling.
- **Annotation safety policy** — missing annotations are reported as `unannotated`; the pipeline does not generate contour-, colour-, or centre-box labels for training.
- **`validator.py`** — shared validation logic used by all converters (bounding box range checks, class ID checks).
- **Likely libraries:** `xml.etree.ElementTree` (VOC), `json` (COCO), `pyyaml` (class map config)

### `src/preprocessing/deduplication.py`
- **Responsibility:** Exact duplicate detection (file hash) and near-duplicate detection (perceptual hashing) — critical for GrapesNet given likely burst-captured frames within subfolders.
- **Key functions:** `compute_hashes(images) -> Dict[path, hash]`, `find_duplicate_groups(hashes, threshold) -> List[DuplicateGroup]`
- **Output:** `reports/deduplication_report.json`
- **Likely libraries:** `imagehash` (pHash/dHash/aHash), `hashlib` (exact-file MD5/SHA1)

### `src/preprocessing/quality_filter.py`
- **Responsibility:** Blur detection (Laplacian variance), annotation-area filtering (zero/tiny boxes), configurable thresholds.
- **Key functions:** `compute_blur_score(image) -> float`, `filter_by_quality(images, thresholds) -> QuarantineResult`
- **Output:** `reports/quality_report.json`
- **Likely libraries:** `opencv-python` (`cv2.Laplacian`)

### `src/preprocessing/splitting.py`
- **Responsibility:** Deterministic, group-aware train/val/test split. Groups = `Dataset-N` subfolder + duplicate-group from Stage 3, to prevent leakage.
- **Key functions:** `split_dataset(images, ratios, seed, groups) -> SplitResult`
- **Output:** `reports/split_report.json`
- **Likely libraries:** `scikit-learn` (`GroupShuffleSplit` or similar), `numpy` (seeded RNG)

### `src/preprocessing/resize.py`
- **Responsibility:** Aspect-ratio-preserving resize + letterbox padding to target size (default 640×640, configurable). Must transform bounding box coordinates identically to the image transform.
- **Key functions:** `letterbox_resize(image, target_size) -> (image, transform_params)`, `transform_bbox(bbox, transform_params) -> bbox`
- **Likely libraries:** `opencv-python` or `Pillow`, `numpy`

### `src/preprocessing/normalization.py`
- **Responsibility:** Documents/enforces normalization behavior. For Ultralytics YOLO, normalization is handled internally at train time — this module's role is mainly to **not** double-normalize, and to record that decision in the run's config snapshot.
- **Key functions:** `get_normalization_policy(framework) -> NormalizationConfig`

### `src/preprocessing/augmentation.py`
- **Responsibility:** Train-only augmentations — horizontal flip, rotation, HSV/brightness/contrast jitter, occlusion simulation, **motion blur** (added specifically for live-video deployment robustness). Must transform bounding boxes consistently with each augmentation.
- **Key functions:** `apply_augmentations(image, bboxes, config) -> (image, bboxes)`
- **Likely libraries:** `albumentations` (handles bbox-aware augmentation directly — recommended over hand-rolled cv2 transforms)

### `src/preprocessing/statistics.py`
- **Responsibility:** Pre/post preprocessing dataset statistics — image counts, objects-per-image (notably different between `Dataset-1`/`1b` vs `Dataset-2`), class distribution, bbox size/area distributions, per-subfolder breakdowns.
- **Output:** `experiments/dataset_vXXX/dataset_statistics.json`

### `src/preprocessing/visualization.py`
- **Responsibility:** Draws bounding boxes on sample images before/after each transformation stage, sampled across all `Dataset-N` subfolders, to visually confirm bbox correctness.
- **Likely libraries:** `opencv-python` or `matplotlib`

### `src/preprocessing/pipeline.py`
- **Responsibility:** Orchestrates all stages in the fixed order, seeds random-number generators, protects existing outputs by default, and runs final image/label integrity validation.

### `scripts/preprocess.py`
- **Responsibility:** CLI entrypoint. Parses `--input`, `--output`, `--config`, `--seed`, `--validate-only`, `--stage`, and calls into `pipeline.py`.
- **Likely libraries:** `argparse` or `click`

---

## 3. Configuration File

### `configs/preprocessing.yaml`
Single source of truth for all tunable parameters — nothing described above should be hardcoded. Expected top-level keys:

```yaml
input_path: "Dataset/"
output_path: "data/processed/"
classes:
  0: grape_bunch
split:
  train: 0.70
  validation: 0.15
  test: 0.15
  seed: 42
resize:
  target_size: 640
quality:
  blur:
    enabled: true
    method: laplacian_variance
    threshold: <configurable_value>
deduplication:
  method: phash
  threshold: <configurable_value>
normalization:
  enabled: false
augmentation:
  horizontal_flip: {enabled: true, probability: 0.5}
  rotation: {enabled: true, degrees: 15}
  hsv: {enabled: true}
  brightness_contrast: {enabled: true}
  occlusion_simulation: {enabled: true}
  motion_blur: {enabled: true}
```

---

## 4. Data Flow Contracts

To keep modules loosely coupled, each stage should communicate via well-defined data structures (not implicit file-path conventions alone):

- `ImageRecord`: `{path, subfolder, width, height, format, corrupted: bool}`
- `AnnotationRecord`: `{image_path, boxes: [{class_id, x_center, y_center, width, height}], source_format}`
- `DuplicateGroup`: `{group_id, image_paths: [...], similarity_score}`
- `SplitAssignment`: `{image_path, split: "train"|"val"|"test", group_id}`

Recommend implementing these as `dataclasses` or `pydantic` models so validation is enforced at each stage boundary, not just at the end.

---

## 5. Suggested Core Libraries

| Purpose | Library |
|---|---|
| Image I/O & corruption checks | `Pillow`, `opencv-python` |
| Perceptual hashing | `imagehash` |
| Bbox-aware augmentation | `albumentations` |
| Config parsing | `pyyaml` |
| Data validation structs | `pydantic` (optional but recommended) |
| Group-aware splitting | `scikit-learn` (`GroupShuffleSplit`) |
| Testing | `pytest` |
| CLI | `argparse` (stdlib) or `click` |

The AI coding agent should pin exact versions in `requirements.txt` once implemented, and record them in `experiments/dataset_vXXX/` for reproducibility, per Section 14 of `PREPROCESSING_PIPELINE.md`.

---

## 6. Extending to a New Dataset

Per the "Most Important Rule" in `PREPROCESSING_PIPELINE.md`, this pipeline should generalize to a new, similarly-structured dataset (e.g., WGISD for cross-dataset validation later in the PhD) by:

1. Pointing `input_path` in `configs/preprocessing.yaml` to the new dataset root.
2. Letting `annotation/format_detector.py` auto-detect the new dataset's annotation format (no code change needed unless it's a genuinely new format, in which case add one converter file).
3. Re-running `scripts/preprocess.py --config configs/preprocessing_<new_dataset>.yaml`.

No changes should be required to `deduplication.py`, `quality_filter.py`, `splitting.py`, `resize.py`, `normalization.py`, or `augmentation.py` for a new dataset with the same general structure (images + one bbox format).
