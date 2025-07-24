import os
from pathlib import Path
from gui_rerank.models.dataset import Dataset

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASETS_DIR = BASE_DIR / 'resources' / 'datasets'
BATCH_NAMES = [f'full_run_{i}' for i in range(1, 6)]
BATCH_SUBDIR = 'rico_filtered_merged'
MERGED_NAME = 'rico_full'


def main():
    print("Loading batch datasets...")
    batch_datasets = []
    for name in BATCH_NAMES:
        batch_path = DATASETS_DIR / name / BATCH_SUBDIR
        print(f"  Loading {batch_path}...")
        ds = Dataset.load(str(batch_path))
        print(ds)
        batch_datasets.append(ds)
    print("Merging datasets with merge_many...")
    merged = Dataset.merge_many(batch_datasets)
    merged.name = MERGED_NAME
    merged.path = str(DATASETS_DIR)
    print(f"Saving merged dataset as {MERGED_NAME} using Dataset.save()...")
    merged.save()
    print(f"Merged dataset saved to {DATASETS_DIR / MERGED_NAME}")

if __name__ == "__main__":
    main() 