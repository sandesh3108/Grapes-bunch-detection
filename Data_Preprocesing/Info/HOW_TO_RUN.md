# How to Run

This guide covers environment setup, running the pipeline, running individual stages, and troubleshooting common errors. It assumes the pipeline has already been implemented by an AI coding agent following `PREPROCESSING_PIPELINE.md` and matches the structure in `TECHNICAL_ARCHITECTURE.md`.

---

## 1. Environment Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Expected core dependencies (see `TECHNICAL_ARCHITECTURE.md` Section 5 for the full list): `Pillow`, `opencv-python`, `imagehash`, `albumentations`, `pyyaml`, `scikit-learn`, `pytest`.

---

## 2. Placing the Dataset

Your local `Dataset/` folder (GrapesNet, with subfolders `Dataset-1`, `Dataset-1b`, `Dataset-2`, `Dataset-3`, `Dataset-4`) should be referenced via `input_path` in `configs/preprocessing.yaml`:

```yaml
input_path: "Dataset/"
```

Do not manually copy/move files into `data/raw/` — the pipeline's ingestion stage handles that mapping. If your dataset folder has a different name or location, just update `input_path` accordingly; nothing else needs to change.

---

## 3. Running the Full Pipeline (Single Command)

Before this command can create a training dataset, every RGB image selected for training needs a reviewed YOLO/VOC/COCO annotation. The default `annotation.mode: require` is deliberate: it stops rather than inventing boxes. Dataset-3 and Dataset-4 depth files are excluded; RGB colour files are retained.

To run the complete 8-stage preprocessing pipeline and generate the ready-to-train dataset with a single command:

```bash
# Navigate to the Data Preprocesing folder and run:
python scripts/preprocess.py --config configs/preprocessing.yaml
```

Each output must use a versioned folder such as `data/processed/v001/`. If that version already exists, the command refuses to overwrite it. Review or archive that dataset, then set `output.overwrite: true` explicitly for a replacement run.

For the full parameter list and operating safeguards, see `CONFIGURATION_REFERENCE.md` and `SECURITY_AND_OPERATIONS.md`.

This single command executes all 8 stages in sequence (Ingestion → Annotation Standardization → Deduplication → Quality Filtering → Leakage-Safe Splitting → 640x640 Letterboxing → Normalization → Train-only Augmentation) and outputs the processed YOLO-ready dataset.

---

## 4. Running Individual Stages (Debugging)

Useful when you want to inspect one stage's output without re-running the whole pipeline:

```bash
python scripts/preprocess.py --stage ingestion
python scripts/preprocess.py --stage validation
python scripts/preprocess.py --stage annotation_standardization
python scripts/preprocess.py --stage deduplication
python scripts/preprocess.py --stage quality_filter
python scripts/preprocess.py --stage split
python scripts/preprocess.py --stage resize
python scripts/preprocess.py --stage augmentation
```

To only validate the raw dataset without writing any processed output (useful as a first sanity check before a full run):

```bash
python scripts/preprocess.py --config configs/preprocessing.yaml --validate-only
```

---

## 5. Useful Flags

| Flag | Purpose |
|---|---|
| `--input` | Override `input_path` from the config |
| `--output` | Override `output_path` from the config |
| `--config` | Path to the YAML config file |
| `--seed` | Override the random seed (for reproducibility experiments) |
| `--validate-only` | Run ingestion/validation only, skip processing |
| `--stage <name>` | Run a single named stage only |

---

## 6. How to Identify if the Dataset is Ready for Training

The dataset is fully preprocessed and ready for YOLO training when the following 4 criteria are met:

1. **`data/processed/` Directory Structure Created**:
   - `data/processed/train/images/` and `data/processed/train/labels/`
   - `data/processed/val/images/` and `data/processed/val/labels/`
   - `data/processed/test/images/` and `data/processed/test/labels/`
   Every `.png`/`.jpg` image inside `images/` has a matching `.txt` label file inside `labels/` with normalized YOLO coordinates (`0 x_center y_center width height`).

2. **Stage Execution Reports Generated**:
   - `reports/ingestion_report.json`: Confirms all raw subfolders were scanned and annotation formats detected.
   - `reports/deduplication_report.json`: Confirms exact/near duplicates were quarantined.
   - `reports/quality_report.json`: Confirms blurred images were removed.
   - `reports/split_report.json`: Confirms the group-aware 70/15/15 train/val/test split ratios.

3. **Experiment & Statistics Verification**:
   - `experiments/dataset_v001/dataset_statistics.json`: Check that image counts, bounding box areas, and objects-per-image metrics are populated.

4. **Visual Bounding Box Inspection**:
   - Inspect `experiments/dataset_v001/visualization_samples/`: Open the generated preview images to visually confirm green bounding boxes tightly frame grape bunches without offset.

Once these 4 steps are verified, you can pass `data/processed/` directly to Ultralytics YOLO training (`yolo train data=...`).

---

## 7. Common Errors and How to Handle Them

### "No valid human-authored annotations were found"
**Cause:** The source images have no accepted YOLO, VOC, or COCO bounding boxes.
**Fix:** This is a safety stop, not a pipeline failure. Create or review annotations in Label Studio, CVAT, LabelImg, or Roboflow, export them in one supported format, and rerun. Do not use colour-contour or centre-box guesses as training labels.

### "Unrecognized annotation format" / conversion failure
**Cause:** A subfolder's annotation files don't match VOC XML, COCO JSON, or YOLO txt.
**Fix:** Open one annotation file manually and check its structure. If it's a legitimate but different format, a new converter needs to be added under `src/preprocessing/annotation/custom_to_yolo.py` — do not force it through an existing converter, as that will silently corrupt bounding boxes.

### "Bounding box outside image boundaries" warnings during ingestion
**Cause:** Either a genuinely malformed annotation, or an annotation referencing the pre-crop/pre-resize image dimensions while a differently-sized image is present.
**Fix:** Check the `ingestion_report.json` for the specific file. Do not auto-clip boxes without reviewing a sample first — clipping can silently mask a real annotation-image mismatch.

### Deduplication flags almost every image as a near-duplicate
**Cause:** The perceptual-hash similarity threshold is too loose for this dataset (vineyard images can look visually similar even when they show different bunches).
**Fix:** Lower the similarity threshold in `configs/preprocessing.yaml` under `deduplication.threshold`, re-run `--stage deduplication`, and check `deduplication_report.json` group sizes before proceeding.

### Class ID validation fails ("class ID X invalid, expected 0")
**Cause:** An annotation file contains a class ID other than `0` for `grape_bunch` — could mean the annotation source used a different class numbering, or a genuinely different class exists in that subfolder.
**Fix:** Check `configs/preprocessing.yaml`'s `classes:` map against what the source annotation actually encodes. Do not silently remap — confirm what the extra class represents first (per `PREPROCESSING_PIPELINE.md`'s "Open Items Requiring User Confirmation").

### Train/val/test split is imbalanced across `Dataset-N` subfolders
**Cause:** Group-aware splitting is working correctly, but one subfolder is much larger/smaller than others, so proportional representation isn't exactly even.
**Fix:** Check `split_report.json`'s per-subfolder breakdown. If the imbalance is problematic for training, consider adjusting split ratios per-subfolder rather than turning off group-awareness (turning it off risks leakage).

### Bounding boxes appear shifted after resize/letterbox
**Cause:** Bbox transform logic isn't accounting for the letterbox padding offset (a common bug in hand-rolled implementations).
**Fix:** Run the visualization step (`visualization.py`) immediately after Stage 6 and inspect samples from each subfolder before proceeding. If boxes are shifted, check that the letterbox function returns and applies the padding offset (not just the scale factor) to every coordinate.

### Augmentation applied to validation or test images
**Cause:** A pipeline bug — augmentation stage running before or across the full dataset instead of only on the `train/` split.
**Fix:** This is a hard rule in `PREPROCESSING_PIPELINE.md` Section 9 — treat this as a blocking bug, not a config tweak. Check that `augmentation.py` is only ever called with the `train/` subset path.

### `pip install` fails on `opencv-python` or `albumentations`
**Cause:** Missing system-level dependencies (common on fresh Linux/WSL environments).
**Fix (Ubuntu/Debian):**
```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
pip install --upgrade pip
pip install opencv-python albumentations
```

### Out-of-memory errors during full-dataset processing
**Cause:** Processing all 11,000+ images (including RGB-D from `Dataset-3`/`Dataset-4`) in memory at once.
**Fix:** Check that the pipeline processes images in batches/streaming rather than loading the entire dataset into memory. If implemented correctly this shouldn't happen — treat repeated OOM errors as a sign the implementation needs batching added, not just a "run with more RAM" workaround.

---

## 8. Reproducing a Previous Run

Every run should snapshot its config:

```text
experiments/dataset_vXXX/preprocessing.yaml
```

To exactly reproduce a past run:

```bash
python scripts/preprocess.py --config experiments/dataset_v001/preprocessing.yaml
```

Given the same dataset, config, seed, and dependency versions, output should be identical — if it isn't, that's a reproducibility bug worth reporting/fixing before relying on results for the PhD write-up.
