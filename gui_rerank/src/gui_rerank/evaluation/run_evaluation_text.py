import os
import ast
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv
import time

from gui_rerank.llm.llm import LLM
from gui_rerank.reranking.llm_reranker_text import LLMRerankerText
from gui_rerank.models.ui_interface_doc import UserInterfaceDocument
from gui_rerank.models.ranked_ui_interface_doc import RankedUserInterfaceDocument

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ANNOTATED_PATH = BASE_DIR / 'resources' / 'evaluation' / 'annotation' / 'goldstandard_llm_general_description_annotations.csv'
RESULTS_DIR = BASE_DIR / 'resources' / 'evaluation' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Set your image directory here (update as needed)
RICO_IMAGE_DIR: Path = Path(r"E:/ZBook Juli 2025/Data/workspace_python/GUI2R/webapp/gui2rapp/staticfiles/resources/combined")


def build_ranked_docs(gui_ids: List[Any], annotations: List[Dict[str, Any]]) -> List[RankedUserInterfaceDocument]:
    docs = []
    for rank, (gui_id, annotation) in enumerate(zip(gui_ids, annotations)):
        img_path = RICO_IMAGE_DIR / f"{gui_id}.jpg"
        # annotation is a dict for this image
        doc = UserInterfaceDocument(id=str(gui_id), filepath=str(img_path), annotation=annotation)
        ranked_doc = RankedUserInterfaceDocument(doc=doc, score=0, rank=rank, source="goldstandard")
        docs.append(ranked_doc)
    return docs


def main():
    # Load annotated goldstandard
    df = pd.read_csv(ANNOTATED_PATH)
    print(df)
    results = []

    # Set up reranker
    #llm = LLM(model_name=LLM.MODEL_GPT_4_1)
    #llm = LLM(model_name=LLM.MODEL_GPT_4_1_NANO)
    llm = LLM(model_name=LLM.MODEL_GPT_4_1_MINI)
    #llm = LLM(model_name=LLM.MODEL_GEMINI_2_5_PRO)
    #llm = LLM(model_name=LLM.MODEL_GEMINI_2_5_FLASH)
    # llm = LLM(model_name=LLM.MODEL_CLAUDE_SONNET_3_7)
    # llm = LLM(model_name=LLM.MODEL_CLAUDE_SONNET_4)
    reranker = LLMRerankerText(llm, mode=LLMRerankerText.MODE_SINGLE_SCORE)
    reranker_type = 'text'

    for counter, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df)), 1):
        query: str = str(row['query'])
        gui_indexes_str: str = str(row['gui_ranking']) if 'gui_ranking' in row else str(row['gui_indexes'])
        gui_ids = ast.literal_eval(gui_indexes_str)
        # The annotation column is a string representation of a list of dicts
        annotation_list = row['llm_general_description_annotations']
        if isinstance(annotation_list, str):
            try:
                annotation_list = ast.literal_eval(annotation_list)
            except Exception:
                annotation_list = [{} for _ in gui_ids]
        # Ensure annotation_list is a list of dicts
        if not (isinstance(annotation_list, list) and all(isinstance(a, dict) for a in annotation_list)):
            annotation_list = [{} for _ in gui_ids]
        # Build docs
        ranked_docs = build_ranked_docs(gui_ids, annotation_list)
        print(f"Ranked docs: {ranked_docs}")
        
        # Print first entry details
        if ranked_docs:
            first_doc = ranked_docs[5]
            print(f"First entry - ID: {first_doc.doc.id}, Annotation: {first_doc.doc.annotation}")
        
        # Rerank with timing
        start_time = time.perf_counter()
        reranked = reranker.rerank(ranked_docs, query, conf_threshold=0, top_k=100)
        end_time = time.perf_counter()
        rerank_time_seconds = end_time - start_time
        #print(f"Reranked: {reranked}")
        reranked_ids = [int(r.doc.id) for r in reranked if r.doc.id is not None]
        id_score_map = {int(r.doc.id): float(r.score) for r in reranked if r.doc.id is not None}
        usage = reranker.last_usage_metadata
        results.append({
            'query': query,
            'gui_ranking': str(reranked_ids),
            'rerank_time_seconds': rerank_time_seconds,
            'input_tokens': usage.get('input_tokens', 0),
            'output_tokens': usage.get('output_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'id_score_map': str(id_score_map),
        })
        print(f"Processed query {counter}/{len(df)} (rerank time: {rerank_time_seconds:.2f}s, input_tokens: {usage.get('input_tokens', 0)}, output_tokens: {usage.get('output_tokens', 0)}, total_tokens: {usage.get('total_tokens', 0)})")

    # Save results
    out_df = pd.DataFrame(results)
    model_name = str(llm.model_name).replace('.', '_')
    temp = str(llm.temperature).replace('.', '_')
    out_name = f"rerank_results_{model_name}_{reranker_type}_temp{temp}.csv"
    out_path = RESULTS_DIR / out_name
    out_df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
