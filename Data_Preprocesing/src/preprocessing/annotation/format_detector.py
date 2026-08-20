"""
Annotation Format Detector.
Inspects subfolders to auto-detect whether annotations are VOC XML, COCO JSON, YOLO TXT, or None.
Runs per-subfolder to accommodate heterogeneous annotation formats across dataset subsets.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def detect_annotation_format(subfolder_path: str) -> str:
    """
    Detects the annotation format in a given directory path.
    Returns one of: 'voc_xml', 'coco_json', 'yolo_txt', 'none', or 'custom'.
    """
    subfolder = Path(subfolder_path)
    if not subfolder.exists() or not subfolder.is_dir():
        return "none"

    xml_files = list(subfolder.rglob("*.xml"))
    json_files = list(subfolder.rglob("*.json"))
    txt_files = [f for f in subfolder.rglob("*.txt") if f.name != "classes.txt" and f.name != "README.txt"]

    # 1. Check for COCO JSON
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "annotations" in data and "images" in data:
                    return "coco_json"
        except Exception:
            pass

    # 2. Check for VOC XML
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            if root.tag == "annotation" or root.find("object") is not None:
                return "voc_xml"
        except Exception:
            pass

    # 3. Check for YOLO TXT
    if txt_files:
        valid_yolo_lines = 0
        total_checked = 0
        for txt_file in txt_files[:20]:  # check up to 20 files
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    for line in lines:
                        parts = line.split()
                        total_checked += 1
                        if len(parts) == 5:
                            try:
                                cls_id = int(parts[0])
                                coords = [float(p) for p in parts[1:]]
                                if all(0.0 <= c <= 1.0 for c in coords):
                                    valid_yolo_lines += 1
                            except ValueError:
                                pass
            except Exception:
                pass
        if total_checked > 0 and (valid_yolo_lines / total_checked) > 0.5:
            return "yolo_txt"

    if xml_files or json_files or txt_files:
        return "custom"

    return "none"


def find_all_images(subfolder_path: str) -> List[Path]:
    """Finds all image files recursively within subfolder_path."""
    images = []
    subfolder = Path(subfolder_path)
    if not subfolder.exists():
        return images
    for root, _, files in os.walk(subfolder):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(Path(root) / file)
    return images
