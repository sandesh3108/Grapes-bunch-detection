"""
CLI Entry Point for Label_Images Data Preprocessing Pipeline.
"""

import sys
import argparse
from pathlib import Path

# Add project root and src to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing.pipeline import LabelImagesPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Label_Images Data Preprocessing Pipeline CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        print(f"Error: Config file not found at '{config_path}'")
        sys.exit(1)

    pipeline = LabelImagesPipeline(str(config_path))
    pipeline.run()

if __name__ == "__main__":
    main()
