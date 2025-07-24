import pandas as pd
import ast
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ANNOTATION_PATH = BASE_DIR / 'resources' / 'evaluation' / 'annotation' / 'goldstandard_llm_general_description_annotations.csv'

def main():
    # Set these indices to inspect a specific instance
    row_index = 81  # <-- set this to the row you want
    image_index = 4  # <-- set this to the image index you want

    df = pd.read_csv(ANNOTATION_PATH)
    if row_index < 0 or row_index >= len(df):
        print(f"Row index {row_index} out of range (0, {len(df)-1})")
        return
    row = df.iloc[row_index]
    query = row['query']
    gui_ranking_str = row['gui_indexes']
    gui_ids = ast.literal_eval(gui_ranking_str)
    annotation_str = row['llm_general_description_annotations']
    try:
        annotation_list = ast.literal_eval(annotation_str)
    except Exception as e:
        print(f"Failed to parse annotation: {e}")
        annotation_list = []
    if image_index < 0 or image_index >= len(gui_ids):
        print(f"Image index {image_index} out of range (0, {len(gui_ids)-1})")
        return
    image_id = gui_ids[image_index]
    annotation = annotation_list[image_index] if image_index < len(annotation_list) else None
    # Print relevance if available
    relevance = None
    if 'relevance' in row and not pd.isna(row['relevance']):
        try:
            relevances = ast.literal_eval(row['relevance'])
            if isinstance(relevances, list) and image_index < len(relevances):
                relevance = relevances[image_index]
        except Exception as e:
            print(f"Failed to parse relevances: {e}")
    print(f"Row index: {row_index}")
    print(f"Query: {query}")
    print(f"Image index: {image_index}")
    print(f"Image id: {image_id}")
    print(f"Annotation: {annotation}")
    print(f"Relevance: {relevance}")

if __name__ == "__main__":
    main()
