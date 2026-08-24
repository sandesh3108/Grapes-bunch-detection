"""
Auto-labeling Script for Grape Bunch Detection.
Predicts bounding boxes using trained YOLO model and saves Pascal VOC XML files.
"""

import sys
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Auto-label images using trained YOLO model and save Pascal VOC XML")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", help="Path to trained YOLO model (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Directory containing images to auto-label")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--class-name", type=str, default="grape_bunch", help="Object class name for Pascal VOC XML")
    return parser.parse_args()

def create_pascal_voc_xml(image_path: Path, boxes, img_width: int, img_height: int, class_name: str) -> str:
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = image_path.parent.name

    filename = ET.SubElement(annotation, "filename")
    filename.text = image_path.name

    path = ET.SubElement(annotation, "path")
    path.text = str(image_path)

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_width)
    height = ET.SubElement(size, "height")
    height.text = str(img_height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for box in boxes:
        # box: xyxy format (xmin, ymin, xmax, ymax)
        xmin_val = int(max(0, min(img_width - 1, box[0])))
        ymin_val = int(max(0, min(img_height - 1, box[1])))
        xmax_val = int(max(0, min(img_width - 1, box[2])))
        ymax_val = int(max(0, min(img_height - 1, box[3])))

        # Skip invalid boxes
        if xmax_val <= xmin_val or ymax_val <= ymin_val:
            continue

        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = class_name

        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(xmin_val)
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(ymin_val)
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(xmax_val)
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(ymax_val)

    # Pretty format XML string
    rough_string = ET.tostring(annotation, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: 'ultralytics' package is not installed. Run 'python -m pip install ultralytics' first.")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found at '{model_path}'")
        sys.exit(1)

    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"Error: Source directory not found at '{source_dir}'")
        sys.exit(1)

    print(f"[Auto-Label] Loading YOLO model: {model_path}")
    model = YOLO(str(model_path))

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions]

    print(f"[Auto-Label] Found {len(all_images)} images in {source_dir}")

    processed_count = 0
    skipped_count = 0

    for img_path in all_images:
        xml_path = img_path.parent / f"{img_path.stem}.xml"
        if xml_path.exists():
            skipped_count += 1
            continue

        # Run inference
        results = model.predict(source=str(img_path), conf=args.conf, verbose=False)
        if not results:
            continue

        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        xml_str = create_pascal_voc_xml(img_path, boxes, img_w, img_h, args.class_name)

        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_str)

        processed_count += 1
        print(f"  [+] Generated XML for: {img_path.name} ({len(boxes)} boxes)")

    print(f"\n[Auto-Label] Finished! Auto-labeled: {processed_count}, Skipped (already had XML): {skipped_count}")

if __name__ == "__main__":
    main()
