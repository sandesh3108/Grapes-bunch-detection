import os
from pathlib import Path

root = Path(r"Dataset/GrapesNet")
if not root.exists():
    root = Path(r"../Dataset/GrapesNet")

print(f"Scanning root: {root.resolve()}")
file_types = {}
non_image_files = []

for p in root.rglob("*"):
    if p.is_file():
        ext = p.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
            non_image_files.append(str(p))

print("File extension summary:", file_types)
print(f"Total non-image files found: {len(non_image_files)}")
if non_image_files:
    print("Sample non-image files:", non_image_files[:20])
