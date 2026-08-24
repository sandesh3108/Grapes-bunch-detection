"""Configuration and filesystem-boundary validation for preprocessing runs."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANAGED_PATH_KEYS = ("output_path", "rejected_path", "reports_path", "experiments_path")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid preprocessing configuration: {message}")


def _managed_path(value: Any, key: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{key} must be a non-empty path")
    candidate = Path(value)
    resolved = (PROJECT_ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Invalid preprocessing configuration: {key} must stay inside {PROJECT_ROOT}") from error
    _require(relative != Path("."), f"{key} cannot be the project root")
    if key == "output_path":
        _require(relative not in {Path("data"), Path("data/processed")},
                 "output_path must be a versioned child of data/processed (for example data/processed/v001)")
    return str(resolved)


def validate_and_normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate all supported parameters before any files are read or written."""
    _require(isinstance(config, dict), "top level must be a mapping")
    normalized = deepcopy(config)

    for key in MANAGED_PATH_KEYS:
        normalized[key] = _managed_path(normalized.get(key), key)

    _require(isinstance(normalized.get("input_path"), str) and normalized["input_path"].strip(),
             "input_path must be a non-empty path")
    classes = normalized.get("classes")
    _require(isinstance(classes, dict) and classes, "classes must be a non-empty mapping")
    for class_id, name in classes.items():
        try:
            numeric_id = int(class_id)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid preprocessing configuration: every class ID must be an integer") from error
        _require(numeric_id >= 0 and isinstance(name, str) and name.strip(),
                 "every class ID must be a non-negative integer with a name")

    annotation = normalized.get("annotation", {})
    _require(annotation.get("mode", "require") in {"require", "manual-later"},
             "annotation.mode must be require or manual-later")
    input_config = normalized.get("input", {})
    modalities = input_config.get("modalities", ["rgb"])
    _require(isinstance(modalities, list) and modalities and set(modalities).issubset({"rgb"}),
             "input.modalities currently supports a non-empty [rgb] list only")
    labels_dir = input_config.get("yolo_labels_dir")
    _require(labels_dir is None or (isinstance(labels_dir, str) and labels_dir.strip()),
             "input.yolo_labels_dir must be a non-empty path when provided")

    split = normalized.get("split", {})
    ratios = [split.get("train"), split.get("validation"), split.get("test")]
    _require(all(isinstance(value, (int, float)) and 0 < value < 1 for value in ratios),
             "split ratios must be numbers strictly between 0 and 1")
    _require(abs(sum(ratios) - 1.0) < 1e-8, "split ratios must sum to 1.0")
    _require(isinstance(split.get("seed"), int) and split["seed"] >= 0, "split.seed must be a non-negative integer")

    resize = normalized.get("resize", {})
    _require(isinstance(resize.get("target_size"), int) and resize["target_size"] > 0,
             "resize.target_size must be a positive integer")
    color = resize.get("padding_color", [114, 114, 114])
    _require(isinstance(color, list) and len(color) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in color),
             "resize.padding_color must be three integers from 0 to 255")

    quality = normalized.get("quality", {})
    _require(quality.get("blur", {}).get("method", "laplacian_variance") == "laplacian_variance",
             "quality.blur.method must be laplacian_variance")
    _require(isinstance(quality.get("blur", {}).get("threshold", 0), (int, float)) and
             quality.get("blur", {}).get("threshold", 0) >= 0,
             "quality.blur.threshold must be a non-negative number")
    _require(float(quality.get("min_box_area_ratio", 0)) > 0 and
             float(quality.get("min_box_area_ratio", 0)) < float(quality.get("max_box_area_ratio", 1)) <= 1,
             "quality box-area ratios must satisfy 0 < min < max <= 1")

    dedup = normalized.get("deduplication", {})
    _require(dedup.get("method", "phash") in {"phash", "dhash", "ahash"},
             "deduplication.method must be phash, dhash, or ahash")
    _require(isinstance(dedup.get("threshold"), int) and 0 <= dedup["threshold"] <= 64,
             "deduplication.threshold must be an integer from 0 to 64")

    for name, options in normalized.get("augmentation", {}).items():
        _require(isinstance(options, dict), f"augmentation.{name} must be a mapping")
        if "probability" in options:
            _require(isinstance(options["probability"], (int, float)) and 0 <= options["probability"] <= 1,
                     f"augmentation.{name}.probability must be between 0 and 1")
    _require(isinstance(normalized.get("output", {}).get("overwrite", False), bool),
             "output.overwrite must be true or false")
    return normalized
