from json import load
import os
import ast
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from gui_rerank.llm.llm import LLM
from gui_rerank.annotator.llm_annotator import LLMAnnotator
from gui_rerank.annotator.config import DEFAULT_SEARCH_DIMENSIONS

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
GOLD_PATH = BASE_DIR / 'resources' / 'evaluation' / 'goldstandard' / 'goldstandard.csv'
ERROR_PATH = BASE_DIR / 'resources' / 'evaluation' / 'annotation' / 'goldstandard_llm_general_description_annotations_errors.csv'
ANNOTATION_OUT_DIR = BASE_DIR / 'resources' / 'evaluation' / 'annotation'
ANNOTATION_OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = ANNOTATION_OUT_DIR / 'goldstandard_llm_general_description_annotations.csv'
RICO_IMAGE_DIR: Path = Path(r"E:/ZBook Juli 2025/Data/workspace_python/GUI2R/webapp/gui2rapp/staticfiles/resources/combined")

# Get only the general_description SearchDimension
GENERAL_DESCRIPTION_DIM = [d for d in DEFAULT_SEARCH_DIMENSIONS if d.name == 'general_description']
if not GENERAL_DESCRIPTION_DIM:
    raise ValueError('general_description SearchDimension not found in DEFAULT_SEARCH_DIMENSIONS')
general_description_dim = GENERAL_DESCRIPTION_DIM[0]


def main():
    # Load goldstandard
    df = pd.read_csv(ERROR_PATH)
    print(df[:1])
    all_annotations = [None] * len(df)

    # If output file exists, load it and skip already annotated rows
    if OUT_PATH.exists():
        print(f"Loading existing annotation file: {OUT_PATH}")
        df_out = pd.read_csv(OUT_PATH)
        if 'llm_general_description_annotations' in df_out.columns:
            prev_annotations = df_out['llm_general_description_annotations'].tolist()
            # If lengths match, use previous annotations
            if len(prev_annotations) == len(df):
                all_annotations = prev_annotations
    
    # Set up LLMAnnotator with only general_description
    llm = LLM(model_name=LLM.MODEL_GPT_4_1)
    annotator = LLMAnnotator(llm=llm, batch_size=5, search_dimensions=[general_description_dim])

    for counter, row in tqdm(df.iterrows(), total=len(df)):
        # Skip if already annotated
        if all_annotations[counter] not in (None, '', [], '[]'):
            continue
        gui_indexes_str = str(row['gui_ranking']) if 'gui_ranking' in row else str(row['gui_indexes'])
        gui_ids = ast.literal_eval(gui_indexes_str)
        image_paths = [str(RICO_IMAGE_DIR / f"{gui_id}.jpg") for gui_id in gui_ids]
        print(f"image_paths: {image_paths}")
        try:
            batch_annotations = annotator.annotate_batch(image_paths)
        except Exception as e:
            print(f"Annotation failed for row {counter}: {e}")
            batch_annotations = [{} for _ in image_paths]
        all_annotations[counter] = batch_annotations
        # Save progress after each row
        df['llm_general_description_annotations'] = all_annotations
        df.to_csv(OUT_PATH, index=False)
        print(f"Saved progress to {OUT_PATH} at row {counter}")

    print(f"Finished. Saved annotated DataFrame to {OUT_PATH}")

if __name__ == "__main__":
    main()
