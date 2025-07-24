import pandas as pd
import ast
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ANNOTATION_PATH = BASE_DIR / 'resources' / 'evaluation' / 'annotation' / 'goldstandard_llm_general_description_annotations_adapted.csv'
ERROR_PATH = BASE_DIR / 'resources' / 'evaluation' / 'annotation' / 'goldstandard_llm_general_description_annotations_errors.csv'

def main():
    df = pd.read_csv(ANNOTATION_PATH)
    not_20 = []
    error_indices = []
    for idx, row in df.iterrows():
        query = row['query']
        annotation_str = row['llm_general_description_annotations']
        try:
            annotation_list = ast.literal_eval(annotation_str)
        except Exception as e:
            print(f"Row {idx}: Failed to parse annotation: {e}")
            annotation_list = []
        count_with_key = sum(1 for d in annotation_list if isinstance(d, dict) and 'general_description' in d)
        print(f"Query: {query}")
        print(f"Number of dicts with 'general_description': {count_with_key}")
        if count_with_key != 20:
            not_20.append((query, count_with_key))
            error_indices.append(idx)
    if not_20:
        print("\nQueries with number of dicts with 'general_description' not equal to 20:")
        for q, c in not_20:
            print(f"Query: {q} | Count: {c}")
        # Save erroneous rows to new DataFrame
        error_df = df.loc[error_indices].copy()
        error_df.to_csv(ERROR_PATH, index=False)
        print(f"\nSaved erroneous rows to {ERROR_PATH}")
    else:
        print("\nAll queries have 20 dicts with 'general_description'.")

if __name__ == "__main__":
    main()
