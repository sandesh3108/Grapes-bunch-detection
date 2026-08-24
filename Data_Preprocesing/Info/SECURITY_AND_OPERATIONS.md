# Security and Operations Guide

## Built-in safeguards

- The raw `Dataset/` path is read only; the pipeline copies rejected samples and never moves or edits source images.
- Managed write locations (`output`, `rejected`, `reports`, and `experiments`) must resolve inside `Data_Preprocesing/`. Paths escaping this directory are rejected before processing.
- The final output must be a versioned child of `data/processed/`; broad targets such as the project root or `data/` cannot be used.
- Existing output is not deleted unless `output.overwrite: true` is explicitly configured.
- Every output image must have a matching label file, and every emitted YOLO row is validated before a run is accepted.
- Missing or invalid annotations do not become guessed training labels.

## Safe run procedure

1. Use a virtual environment and install the listed dependencies.
2. Keep the raw dataset outside generated output directories.
3. Run validation first: `python scripts/preprocess.py --config configs/preprocessing.yaml --validate-only`.
4. Review ingestion and annotation reports, then correct annotation issues.
5. Create a new versioned output path for the full run.
6. Review `final_integrity_report.json`, statistics, and visual samples before training.
7. Retain configuration snapshots and reports with every experiment result.

## Operational limits

- Perceptual deduplication is computationally expensive on large datasets. Run it on local storage, not a network-synced folder when possible.
- The current pipeline supports an RGB detection baseline only. RGB-D fusion requires a separate, explicitly designed data contract and model input path.
- The configuration protects filesystem destinations; it does not replace operating-system permissions, backups, or antivirus controls.

## Incident response

If a run fails, do not reuse a partial output. Preserve the reports, choose a new output version, fix the reported cause, and rerun. Never solve a validation failure by disabling the final integrity check.
