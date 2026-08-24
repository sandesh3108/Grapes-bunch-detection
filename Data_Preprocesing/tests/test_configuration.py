from pathlib import Path

import pytest
import yaml

from src.preprocessing.configuration import PROJECT_ROOT, validate_and_normalize_config


def load_default_config():
    return yaml.safe_load((PROJECT_ROOT / "configs" / "preprocessing.yaml").read_text(encoding="utf-8"))


def test_default_configuration_is_valid_and_managed_paths_are_absolute():
    config = validate_and_normalize_config(load_default_config())
    assert Path(config["output_path"]).is_relative_to(PROJECT_ROOT)
    assert config["output_path"].endswith("data\\processed\\v001")


def test_configuration_rejects_unsafe_output_target():
    config = load_default_config()
    config["output_path"] = ".."
    with pytest.raises(ValueError, match="must stay inside"):
        validate_and_normalize_config(config)


def test_configuration_rejects_invalid_split_ratios():
    config = load_default_config()
    config["split"]["test"] = 0.2
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_and_normalize_config(config)
