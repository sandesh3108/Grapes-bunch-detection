# Dataset Preprocessing Pipeline — Specification for AI Coding Agent

**Project:** Grape Bunch Detection (PhD Project)
**Task type:** Single-class object detection
**Target deployment:** Trained on static images, deployed on live video (frame-by-frame inference + tracking layer added later, out of scope for this preprocessing pipeline)
**Intended model family:** YOLO (exact version — YOLOv11 or YOLOv12 — not finalized yet; pipeline output must be YOLO-format-compatible regardless of exact version)

This document is the **single source of truth** for building the preprocessing pipeline. An AI coding agent should be able to read this file top to bottom and implement the entire system without further clarification. If something in this document is ambiguous, the agent should flag it and ask — it must not silently assume dataset-specific behavior that isn't stated here.

---

## 0. Dataset Context (must be respected by the pipeline)

- **Dataset name:** GrapesNet
- **Local folder name:** `Dataset/` (this is the raw input root — do not rename it, reference it as configurable input path)
- **Internal structure:** Subfolders per GrapesNet subset, e.g.:
  ```text
  Dataset/
  ├── Dataset-1/      # single grape cluster, artificial background
  ├── Dataset-1b/     # single grape cluster, real background
  ├── Dataset-2/      # multiple clusters per image, real background
  ├── Dataset-3/      # RGB-D images, natural vineyard environment
  └── Dataset-4/      # RGB-D, single bunch, coral background, weight-labeled
  ```
- **Class configuration:** Single class only.
  ```yaml
  classes:
    0: grape_bunch
  ```
- **Annotation format: UNKNOWN / NOT YET CONFIRMED.** This is explicitly called out, not assumed. The pipeline's ingestion stage must **auto-detect** the annotation format present in each subfolder rather than hardcoding one. Supported formats to detect and handle, in this priority order:
  1. YOLO `.txt` (one file per image, `class_id x_center y_center width height`, normalized)
  2. Pascal VOC `.xml` (one file per image)
  3. COCO `.json` (single file covering multiple images)
  4. **No annotation files present** — in this case, the image must be logged as "unannotated" in the ingestion report, NOT deleted or skipped silently. The pipeline should support a `--annotation-mode manual-later` flag that allows proceeding with image-only processing (resize/dedupe/quality-filter) so the user can annotate afterward (e.g., via LabelImg/CVAT/Roboflow) and re-run annotation standardization independently.
  5. Each `Dataset-N/` subfolder may use a **different** annotation format from the others (since these are separate sub-datasets released together) — the detector must run per-subfolder, not assume one global format for the whole `Dataset/` root.
- **Source/group awareness:** Each `Dataset-N/` subfolder must be treated as a distinct **source group** for leakage-prevention purposes during splitting (see Stage 5). Images from `Dataset-2` (multi-cluster, real background) must not be allowed to "balance out" against `Dataset-1` (single-cluster, artificial background) in a way that causes near-identical scenes to leak across train/val/test — group-aware splitting must operate at the subfolder level at minimum, and at the perceptual-hash duplicate-group level within subfolders.
- **RGB-D subfolders (`Dataset-3`, `Dataset-4`):** These contain depth data in addition to RGB. For this phase of the project (bunch detection), **only the RGB channel is required**. The pipeline should ingest RGB images from these subfolders normally, and depth data should be preserved but not processed by this pipeline (flag as out-of-scope, do not discard the files, do not silently convert/merge into RGB).
- **`Dataset-4` weight labels (height/width/weight in cm/kg):** Out of scope for the bunch-detection task. Do not attempt to use these as detection targets. Preserve untouched in `data/raw/`.

If any of the above turns out to be inaccurate once the agent actually inspects the real folder contents, **stop and report the discrepancy** rather than silently adapting around it.

---

## 1. Overall Pipeline

Implement the preprocessing pipeline in the following order:

```text
Raw Dataset
    ↓
1. Ingestion & Validation  (includes annotation-format auto-detection, see Section 0)
    ↓
2. Annotation Standardization
    ↓
3. Deduplication
    ↓
4. Quality Filtering
    ↓
5. Dataset Split
    ↓
6. Resize / Letterbox
    ↓
7. Normalization
    ↓
8. Train-only Augmentation
    ↓
Final Dataset
```

**IMPORTANT:**
- Do NOT perform augmentation before the dataset split.
- The validation and test datasets must never contain augmented versions of training images.
- The test dataset must remain untouched by training augmentation.

---

## 2. Stage 1 — Ingestion and Validation

Create a dedicated ingestion and validation module.

The module must:

- Scan the complete input dataset recursively (all `Dataset-N/` subfolders).
- Identify supported image formats.
- Identify annotation files, **auto-detecting format per subfolder** as described in Section 0.
- Match images and annotations correctly.
- Verify that every annotated image has a valid annotation file where required.
- Detect images without annotations (log, do not delete — see Section 0, item 4).
- Detect annotations without corresponding images.
- Detect corrupted or unreadable images.
- Detect invalid image dimensions.
- Detect unsupported file formats.
- Detect empty annotation files.
- Validate annotation syntax.
- Validate class IDs (must be `0` only, given single-class `grape_bunch` — flag any other class ID as invalid).
- Validate bounding-box coordinates.
- Detect bounding boxes outside valid image boundaries.
- Detect malformed annotation rows.
- Record, per subfolder, which annotation format was auto-detected (or "none detected").

Do not silently delete invalid files.

Generate a validation report:

```text
reports/
└── ingestion_report.json
```

The report should include:

- Total images (overall, and broken down per `Dataset-N` subfolder)
- Valid images
- Corrupted images
- Missing annotations
- Orphan annotations
- Invalid annotations
- Empty annotations
- Invalid bounding boxes
- Unsupported files
- Detected annotation format per subfolder

---

## 3. Stage 2 — Annotation Standardization

The preprocessing pipeline must not depend on the original dataset annotation format.

Support conversion from formats such as:

```text
Pascal VOC XML
COCO JSON
YOLO txt (pass-through / validate only)
Manual/custom annotations
```

into a unified YOLO format:

```text
class_id x_center y_center width height
```

All coordinates must be normalized according to YOLO requirements.

Example:

```text
0 0.512 0.431 0.245 0.318
```

Create a dedicated annotation conversion module.

Recommended structure:

```text
src/
└── preprocessing/
    └── annotations/
        ├── voc_to_yolo.py
        ├── coco_to_yolo.py
        ├── yolo_passthrough_validator.py
        ├── custom_to_yolo.py
        ├── format_detector.py        # NEW: auto-detects format per subfolder (Section 0)
        └── validator.py
```

The conversion process must:

- Preserve class information (single class: `grape_bunch` → id `0`).
- Maintain image-to-label pairing.
- Validate converted labels.
- Log conversion failures.
- Never silently discard annotations.

Create a class mapping configuration instead of hardcoding class IDs throughout the code:

```yaml
classes:
  0: grape_bunch
```

---

## 4. Stage 3 — Deduplication

Detect exact and near-duplicate images.

Use:

- File/hash comparison for exact duplicates.
- Perceptual hashing for visually similar images (relevant here — GrapesNet subfolders may contain burst-captured or closely-spaced frames of the same cluster).

Examples of perceptual hashing methods may include:

- pHash
- dHash
- aHash

The threshold must be configurable.

Do not automatically delete duplicates without recording them.

Generate a report:

```text
reports/
└── deduplication_report.json
```

The report should contain:

- Number of exact duplicates
- Number of near duplicates
- Duplicate groups
- Images retained
- Images removed/quarantined
- Similarity threshold used
- **Duplicate groups broken down per `Dataset-N` subfolder** (helps confirm whether duplication is within a subfolder or, unexpectedly, across subfolders)

**IMPORTANT:** Deduplication must happen BEFORE train/validation/test splitting. This prevents near-identical images from appearing across different dataset splits.

---

## 5. Stage 4 — Quality Filtering

Create a reusable image-quality validation module.

Detect problematic images using configurable thresholds.

At minimum support:

### Blur detection

Use an appropriate blur metric such as Laplacian variance.

Do not hardcode the threshold.

```yaml
quality:
  blur:
    enabled: true
    method: laplacian_variance
    threshold: configurable_value
```

### Annotation-area filtering

Detect:

- Images with zero annotations
- Images with extremely small annotation areas
- Invalid bounding boxes
- Abnormally large or small objects where appropriate

Do not automatically remove potentially useful samples merely because they are unusual.

Use configurable thresholds and generate a report:

```text
reports/
└── quality_report.json
```

Each rejected/quarantined image should include a reason.

---

## 6. Stage 5 — Dataset Splitting

Split the cleaned dataset into:

```text
70% train
15% validation
15% test
```

Make these ratios configurable.

The split must be deterministic using a configurable random seed.

```yaml
split:
  train: 0.70
  validation: 0.15
  test: 0.15
  seed: 42
```

The split must avoid data leakage.

Since the dataset contains **multiple sources** (the `Dataset-1`, `Dataset-1b`, `Dataset-2`, `Dataset-3`, `Dataset-4` subfolders, each with different capture conditions):

- Do NOT randomly distribute related images across train/validation/test without regard to source.
- Use **group-aware splitting**: at minimum, stratify by `Dataset-N` subfolder so each split contains a representative mix of all subsets (not, e.g., all of `Dataset-2` ending up only in train).
- Within each subfolder, also respect any burst/duplicate-groups identified in Stage 3 — keep them together in the same split.
- Prevent near-identical frames from appearing in different splits.

The final report must show:

```text
Train:
    images:
    annotations:
    classes:

Validation:
    images:
    annotations:
    classes:

Test:
    images:
    annotations:
    classes:
```

Also report class distribution for every split, and the per-subfolder (`Dataset-N`) composition of each split.

---

## 7. Stage 6 — Resize and Letterbox

Resize images to the model input size.

```text
640 × 640
```

The input size must be configurable (so it can be changed later depending on the final YOLOv11/v12 variant chosen).

Do NOT stretch images directly to the target dimensions.

Use aspect-ratio-preserving resize with letterboxing/padding:

```text
Original image
      ↓
Preserve aspect ratio
      ↓
Resize
      ↓
Padding
      ↓
640 × 640
```

Bounding boxes must be transformed consistently with the image. This is critical.

After resizing/letterboxing, validate that every bounding box still corresponds correctly to the object.

Create tests for bounding-box transformation.

---

## 8. Stage 7 — Normalization

Normalization must depend on the actual ML framework.

Do not blindly modify and save normalized images.

If using Ultralytics YOLO (YOLOv11/v12), verify how the framework performs:

- Pixel scaling
- Channel conversion
- Tensor conversion
- Normalization

Avoid performing the same normalization twice — Ultralytics YOLO handles normalization internally at training time, so this pipeline should generally **not** bake normalization into saved image files; it should document this decision instead of duplicating the operation.

```yaml
normalization:
  enabled: false   # handled internally by Ultralytics YOLO — see note below
  scale: null
  mean: null
  std: null
```

If normalization is handled internally by the training framework (as it is for Ultralytics YOLO), document that clearly in the pipeline output/config rather than duplicating the operation.

---

## 9. Stage 8 — Train-only Augmentation

Augmentation must be applied ONLY to the training dataset.

Never apply training augmentation to:

```text
validation
test
```

The purpose of augmentation is to simulate realistic variations expected during deployment — for this project specifically: **variable outdoor vineyard lighting, occlusion from leaves/other bunches, and (per the live-video deployment discussion) motion blur from a moving camera/robotic arm.**

Augmentations to include, and why:

```yaml
augmentation:
  horizontal_flip:
    enabled: true
    probability: 0.5
    reason: "Vineyard rows/bunches have no fixed orientation dependency"

  rotation:
    enabled: true
    degrees: 15
    reason: "Camera angle varies during handheld/robotic capture"

  hsv:
    enabled: true
    reason: "Simulates varying outdoor illuminance noted in GrapesNet capture conditions"

  brightness_contrast:
    enabled: true
    reason: "Same as above — lighting variability is a known challenge for this dataset/task"

  occlusion_simulation:
    enabled: true
    reason: "Grape bunches are frequently leaf-occluded; literature review identified occlusion robustness as a key open problem"

  motion_blur:
    enabled: true
    reason: "Deployment target is live video from a moving camera/robotic arm, not just static images — training must account for this even though the source dataset is static images"
```

Use only augmentations that make sense for the actual problem. Do NOT randomly add augmentations just to increase the number of transformations.

For each augmentation, document: Augmentation / Purpose / Probability / Parameter range / Reason for inclusion (as above).

Augmentation must also transform bounding boxes correctly.

---

## 10. Data Leakage Prevention

This is a mandatory requirement.

The pipeline must prevent leakage caused by:

- Duplicate images
- Near-duplicate images
- Burst frames
- Video frames (relevant later, if/when live-video-captured data is added to training)
- Same scene captured multiple times
- Same source subfolder (`Dataset-N`) appearing disproportionately across splits
- Preprocessing fitted using validation/test data
- Augmented copies appearing in validation/test

The final pipeline should enforce:

```text
Raw
 ↓
Deduplicate
 ↓
Clean
 ↓
Group-aware split (grouped by Dataset-N subfolder + duplicate-group)
 ↓
Train augmentation
```

---

## 11. Dataset Directory Structure

Use a clean structure such as:

```text
data/
├── raw/                          # mirrors the original Dataset/ folder — never modified
│   ├── Dataset-1/
│   ├── Dataset-1b/
│   ├── Dataset-2/
│   ├── Dataset-3/
│   └── Dataset-4/
│
├── intermediate/
│
├── processed/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   │
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   │
│   └── test/
│       ├── images/
│       └── labels/
│
└── rejected/
    ├── corrupted/
    ├── invalid_annotations/
    ├── duplicates/
    ├── low_quality/
    └── unannotated/               # NEW: images with no detected annotation (Section 0)
```

Do not modify the original files inside `data/raw/`. The user's actual local folder is named `Dataset/` — the agent should either symlink/copy it into `data/raw/` per this structure, or make the raw input path fully configurable and document clearly which convention was chosen.

---

## 12. Configuration

All important preprocessing parameters must be configurable. Do not hardcode values such as:

```text
640
70/15/15
blur threshold
duplicate threshold
rotation angle
augmentation probability
random seed
input dataset path ("Dataset/")
class map (grape_bunch : 0)
```

Use configuration files:

```text
configs/
└── preprocessing.yaml
```

---

## 13. Logging and Reports

Every major stage must produce useful logs.

Example:

```text
[INGESTION] 12,450 images discovered across 5 subfolders (Dataset-1, Dataset-1b, Dataset-2, Dataset-3, Dataset-4)
[INGESTION] Annotation format detected: Dataset-1 = none, Dataset-2 = VOC XML, ...
[VALIDATION] 12,312 valid images
[DEDUPLICATION] 87 duplicate groups detected
[QUALITY] 42 images quarantined
[SPLIT] Train: 8,527
[SPLIT] Validation: 1,830
[SPLIT] Test: 1,826
```

Generate machine-readable reports:

```text
reports/
├── ingestion_report.json
├── annotation_report.json
├── deduplication_report.json
├── quality_report.json
├── split_report.json
└── preprocessing_summary.json
```

---

## 14. Reproducibility

The preprocessing pipeline must produce the same result when executed with the same:

- Dataset
- Configuration
- Random seed
- Software/dependency versions

Store the preprocessing configuration used for each dataset version:

```text
experiments/
└── dataset_v001/
    ├── preprocessing.yaml
    ├── dataset_statistics.json
    └── preprocessing_summary.json
```

---

## 15. Validation After Every Major Stage

Do not assume preprocessing succeeded. Validate the output after each stage:

```text
Ingestion
    ↓
Validation
    ↓
Check
    ↓
Annotation conversion
    ↓
Check
    ↓
Deduplication
    ↓
Check
    ↓
Quality filtering
    ↓
Check
    ↓
Split
    ↓
Check
    ↓
Resize/letterbox
    ↓
Check
    ↓
Augmentation
    ↓
Final validation
```

For object detection, create visualization samples showing:

```text
Image + bounding boxes + class labels
```

before and after important transformations. This is mandatory for verifying that bounding boxes were not corrupted. Include at least a handful of samples from **each `Dataset-N` subfolder**, since transformation correctness may vary by image size/aspect ratio across subsets.

---

## 16. Dataset Statistics

Generate statistics before and after preprocessing. Include:

- Total images
- Image dimensions
- Number of annotations
- Number of classes (should be exactly 1: `grape_bunch`)
- Objects per image (important here — `Dataset-2` has multiple clusters/image, `Dataset-1`/`1b` have one — report this per subfolder)
- Class distribution
- Bounding-box width/height distribution
- Bounding-box area distribution
- Train/validation/test distribution
- Number of rejected images
- Number of duplicates
- Number of corrupted images

Generate visualizations where useful (e.g., objects-per-image histogram, bounding-box size distribution, per-subfolder image counts).

---

## 17. CLI Interface

The preprocessing pipeline should be executable from the command line:

```bash
python scripts/preprocess.py --config configs/preprocessing.yaml
```

Support useful options such as:

```bash
--input
--output
--config
--seed
--validate-only
--stage
```

Allow individual stages to be executed when debugging:

```bash
python scripts/preprocess.py --stage validation
python scripts/preprocess.py --stage deduplication
python scripts/preprocess.py --stage split
```

---

## 18. Modular Implementation

Do NOT create a single `preprocess.py` containing thousands of lines.

Instead use modules such as:

```text
src/preprocessing/
├── ingestion.py
├── validation.py
├── annotation/
│   ├── format_detector.py
│   ├── voc_to_yolo.py
│   ├── coco_to_yolo.py
│   ├── yolo_passthrough_validator.py
│   └── validator.py
├── deduplication.py
├── quality_filter.py
├── splitting.py
├── resize.py
├── normalization.py
├── augmentation.py
├── statistics.py
├── visualization.py
└── pipeline.py
```

The main pipeline should orchestrate these modules rather than implementing all logic itself.

---

## 19. Testing

Create tests for:

- Image validation
- Annotation validation
- Annotation format auto-detection (Section 0/2 — test against synthetic VOC, COCO, YOLO, and no-annotation cases)
- Annotation conversion
- Bounding-box conversion
- Deduplication
- Dataset splitting (including group-aware/subfolder-aware logic)
- Letterbox transformation
- Bounding-box transformation
- Configuration loading
- Augmentation
- Final dataset integrity

Especially test that:

```text
Image transformation
        +
Bounding-box transformation
```

remain geometrically consistent.

---

## 20. Final Acceptance Criteria

The preprocessing system is complete only when:

- Raw data (`Dataset/` / `data/raw/`) remains unchanged.
- All images are validated.
- All annotations are validated.
- Annotation format has been auto-detected and documented per subfolder.
- All annotations are converted to the required unified YOLO format.
- Exact duplicates are handled.
- Near duplicates are handled.
- Poor-quality samples are reported/quarantined.
- Unannotated images are reported/quarantined separately, not silently dropped.
- Dataset leakage is prevented, including subfolder-level (`Dataset-N`) group-aware splitting.
- Dataset is split deterministically (70/15/15, configurable, seeded).
- Resize preserves aspect ratio.
- Letterboxing is correctly implemented.
- Bounding boxes remain correct after transformations.
- Normalization behavior is documented and compatible with Ultralytics YOLO (no double-normalization).
- Augmentation (including motion blur, for future video-deployment robustness) is applied only to training data.
- Validation/test data remain unaugmented.
- Dataset statistics are generated, including per-`Dataset-N`-subfolder breakdowns.
- Preprocessing reports are generated.
- Configuration is reproducible.
- Pipeline can be executed from the command line.
- Tests pass.
- Sample visualizations confirm correct bounding boxes, including samples from every subfolder.

---

## Most Important Rule

Do not optimize only for "getting the GrapesNet dataset ready." Build the preprocessing system as a **reusable dataset-engineering pipeline** that can accept another compatible image/annotation dataset (e.g., WGISD, or any similar vineyard/fruit-detection dataset used later for cross-dataset generalization testing, per the PhD methodology) with minimal code changes.

```text
INGEST
  ↓
VALIDATE
  ↓
STANDARDIZE
  ↓
DEDUPLICATE
  ↓
QUALITY FILTER
  ↓
SPLIT
  ↓
RESIZE / LETTERBOX
  ↓
NORMALIZE
  ↓
TRAIN-ONLY AUGMENTATION
  ↓
VALIDATE AGAIN
  ↓
READY FOR TRAINING
```

Never allow augmentation, duplicate images, or information from validation/test data to contaminate the training/validation/test boundaries.

---

## Open Items Requiring User Confirmation (do not resolve silently)

An AI coding agent implementing this spec should stop and ask the user if any of the following turn out to be true once it inspects the real `Dataset/` folder:

1. The actual annotation format found doesn't match any of VOC/COCO/YOLO (e.g., a proprietary format).
2. Any subfolder contains a class label other than a single `grape_bunch` class.
3. Image counts per subfolder differ drastically from what's expected from the published GrapesNet paper (11,000+ images total across 5 subsets) — could indicate an incomplete download.
4. `Dataset-3`/`Dataset-4` RGB-D files are in a format the agent doesn't recognize (e.g., non-standard depth encoding).
