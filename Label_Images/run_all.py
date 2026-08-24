"""
Master Runner Script: Execute Full Grape Bunch Detection Pipeline end-to-end.
"""

import sys
import subprocess
from pathlib import Path

def run_step(step_name: str, command: list):
    print(f"\n==================================================")
    print(f"  [STEP] {step_name}")
    print(f"==================================================")
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"❌ Step '{step_name}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ Step '{step_name}' completed successfully.")

def main():
    script_dir = Path(__file__).resolve().parent
    python_exe = sys.executable

    print(f"Using Python executable: {python_exe}")
    print(f"Working Directory: {script_dir}")

    # Step 1: Install Requirements
    req_file = script_dir / "requirements.txt"
    if req_file.exists():
        run_step("Install Dependencies from requirements.txt", [
            python_exe, "-m", "pip", "install", "-r", str(req_file)
        ])

    # Step 2: Data Preprocessing Pipeline
    preprocess_script = script_dir / "scripts" / "preprocess.py"
    config_file = script_dir / "configs" / "preprocessing.yaml"
    run_step("Data Preprocessing & Split Generation", [
        python_exe, str(preprocess_script), "--config", str(config_file)
    ])

    # Step 2: Ensure Ultralytics is installed
    try:
        import ultralytics
        print("Ultralytics package is installed.")
    except ImportError:
        print("Installing ultralytics...")
        run_step("Install Ultralytics", [python_exe, "-m", "pip", "install", "ultralytics"])

    # Step 3: Train YOLO Model
    train_script = script_dir / "scripts" / "train.py"
    data_yaml = script_dir / "data" / "processed" / "v001" / "data.yaml"
    run_step("YOLO Model Training (100 Epochs)", [
        python_exe, str(train_script),
        "--data", str(data_yaml),
        "--model", "yolo11n.pt",
        "--epochs", "20",
        "--batch", "8",
        "--imgsz", "640"
    ])

    # Step 4: Auto-Label Remaining Unlabeled Images with Pascal VOC XML
    best_weights = script_dir / "runs" / "detect" / "train" / "weights" / "best.pt"
    dataset_source = script_dir.parent / "Dataset" / "GrapesNet"
    auto_label_script = script_dir / "scripts" / "auto_label.py"

    if dataset_source.exists() and best_weights.exists():
        run_step("Auto-label Unlabeled Images (Pascal VOC XML)", [
            python_exe, str(auto_label_script),
            "--model", str(best_weights),
            "--source", str(dataset_source),
            "--conf", "0.50"
        ])
    else:
        print(f"Skipping auto-labeling: Dataset path ({dataset_source}) or model weights ({best_weights}) not found.")

    print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
