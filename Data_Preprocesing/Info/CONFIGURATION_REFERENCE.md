# Configuration Reference

The canonical file is `configs/preprocessing.yaml`. The command validates it before scanning data or writing output. Invalid values exit with status code 2.

| Parameter | Valid values | Effect |
|---|---|---|
| `input_path` | Non-empty path | Read-only location of raw source imagery. It may be outside the project. |
| `output_path` | A versioned child of `data/processed/` | Final YOLO directory, e.g. `data/processed/v001/`. |
| `rejected_path`, `reports_path`, `experiments_path` | Paths inside `Data_Preprocesing/` | Managed outputs; parent-directory traversal and external locations are rejected. |
| `annotation.mode` | `require`, `manual-later` | `require` blocks a training output if annotations are missing. `manual-later` records the audit and stops. |
| `input.modalities` | `[rgb]` | RGB baseline only; depth data is not processed. |
| `input.exclude_generated_variants` | Boolean | Excludes known published flip/warp derivative names. |
| `output.overwrite` | Boolean | Must be `true` to replace an existing versioned output directory. |
| `classes` | Non-empty `{id: name}` mapping | Allowed YOLO classes. GrapesNet baseline uses only `0: grape_bunch`. |
| `split.train`, `validation`, `test` | Each `(0, 1)`, total exactly `1.0` | Deterministic train/validation/test ratios. |
| `split.seed`, `reproducibility.seed` | Non-negative integers | Seeds split and image-processing randomness. |
| `resize.target_size` | Positive integer | Square letterbox target size. |
| `resize.padding_color` | Three integers `[0..255]` | Letterbox BGR padding colour. |
| `quality.blur.method` | `laplacian_variance` | Current supported blur metric. |
| `quality.blur.threshold` | Non-negative number | Images below this blur score are quarantined. Tune after reviewing samples. |
| `quality.min_box_area_ratio`, `max_box_area_ratio` | `0 < min < max <= 1` | Rejects implausibly small/large normalized boxes. |
| `deduplication.method` | `phash`, `dhash`, `ahash` | Perceptual image-hash algorithm. |
| `deduplication.threshold` | Integer `0..64` | Maximum Hamming distance for near-duplicates; lower is stricter. |
| `augmentation.*.probability` | Number `0..1` | Probability of the named train-only transform. |

## Recommended operating pattern

Create a new output/experiment version per meaningful parameter or annotation revision, for example `v002`. Do not overwrite `v001` until its reports and visual samples have been reviewed.
