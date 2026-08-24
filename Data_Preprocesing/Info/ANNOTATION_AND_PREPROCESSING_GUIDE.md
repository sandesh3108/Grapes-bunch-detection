# Annotation and Preprocessing Guide

## Current dataset state

The checked GrapesNet copy contains RGB and RGB-D imagery but no verified object-detection annotations. It is therefore not a valid YOLO training dataset yet. The preprocessing pipeline now records those images as `unannotated` and refuses to fabricate bounding boxes.

## Required annotation workflow

1. Annotate RGB colour images only. Do not annotate or train on `*_Depth.png` files for the RGB baseline.
2. Use one class: `grape_bunch` with class ID `0`.
3. Export one supported format: YOLO TXT, Pascal VOC XML, or COCO JSON.
4. Review a stratified sample from every source subset and every capture condition before export.
5. Place annotations beside their matching source images, or preserve COCO `file_name` paths relative to the source subset.
6. Run `python scripts/preprocess.py --config configs/preprocessing.yaml --validate-only` first. Resolve every invalid or empty annotation before a full run.

## Safety rules enforced by the pipeline

- Unannotated images never receive guessed boxes and do not enter the YOLO dataset.
- RGB-D depth maps are preserved in the raw source tree but excluded from RGB preprocessing.
- Known published flip/warp variants are excluded by default; their source provenance is also represented in split grouping.
- Existing processed datasets are never removed unless `output.overwrite: true` is set explicitly.
- The final dataset is checked for image/label pairing, YOLO syntax, valid class IDs, and in-bounds boxes.

## Before model training

Confirm that `reports/final_integrity_report.json` has `"valid": true`, inspect rendered boxes from each split, and create a YOLO `data.yaml` that references the final processed train/val/test directories. Keep the test set untouched until final evaluation.
