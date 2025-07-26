import json
import shutil
from pathlib import Path

# Paths
ANNOTATIONS_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / 'rico' / 'dataset' / 'annotations.json'
SRC_IMAGE_DIR = Path(r"E:/ZBook Juli 2025/Data/workspace_python/GUI2R/webapp/gui2rapp/staticfiles/resources/combined")
DST_IMAGE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / 'rico' / 'images'

# Ensure destination directory exists
DST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Load annotation keys
with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

num_copied = 0
num_skipped = 0
for key in annotations.keys():
    image_filename = f"{key}.jpg"
    src_path = SRC_IMAGE_DIR / image_filename
    dst_path = DST_IMAGE_DIR / image_filename
    if not src_path.exists():
        print(f"Source image does not exist: {src_path}")
        num_skipped += 1
        continue
    shutil.copy2(src_path, dst_path)
    print(f"Copied {src_path} -> {dst_path}")
    num_copied += 1

print(f"Done. Copied {num_copied} images, skipped {num_skipped}.") 