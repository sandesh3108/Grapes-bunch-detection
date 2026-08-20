"""
CLI Entry Point for GrapesNet Data Preprocessing Pipeline.
Provides command-line options for running full pipeline or individual stages.
"""

import sys
import argparse
import yaml
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing.pipeline import PreprocessingPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="GrapesNet Modular Data Preprocessing Pipeline CLI"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Override input_path from configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output_path from configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for dataset splitting",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run dataset scanning and validation only, skip processing",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "ingestion",
            "validation",
            "deduplication",
            "quality_filter",
            "split",
            "resize",
            "augmentation",
        ],
        default=None,
        help="Run up to a single specific stage for debugging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        # Search relative to project root
        config_path = project_root / args.config
        if not config_path.exists():
            print(f"Error: Config file not found at '{args.config}'")
            sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.input:
        config["input_path"] = args.input
    if args.output:
        config["output_path"] = args.output
    if args.seed is not None:
        config["split"]["seed"] = args.seed

    pipeline = PreprocessingPipeline(config)
    pipeline.run(validate_only=args.validate_only, stage=args.stage)


if __name__ == "__main__":
    main()
