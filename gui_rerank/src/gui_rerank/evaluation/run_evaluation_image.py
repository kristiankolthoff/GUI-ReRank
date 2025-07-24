import os
import ast
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import time
from tqdm import tqdm

from gui_rerank.llm.llm import LLM
from gui_rerank.reranking.llm_reranker_image import LLMRerankerImage
from gui_rerank.models.ui_interface_doc import UserInterfaceDocument
from gui_rerank.models.ranked_ui_interface_doc import RankedUserInterfaceDocument

load_dotenv()

# Paths
gold_path: Path
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
gold_path = BASE_DIR / 'resources' / 'evaluation' / 'goldstandard' / 'goldstandard.csv'
RICO_IMAGE_DIR: Path = Path(r"E:/ZBook Juli 2025/Data/workspace_python/GUI2R/webapp/gui2rapp/staticfiles/resources/combined")
RESULTS_DIR: Path = BASE_DIR / 'resources' / 'evaluation' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_ranked_docs(gui_ids: List[Any]) -> List[RankedUserInterfaceDocument]:
    docs = []
    for rank, gui_id in enumerate(gui_ids):
        img_path = RICO_IMAGE_DIR / f"{gui_id}.jpg"
        doc = UserInterfaceDocument(id=str(gui_id), filepath=str(img_path), annotation=None)
        ranked_doc = RankedUserInterfaceDocument(doc=doc, score=0, rank=rank, source="goldstandard")
        docs.append(ranked_doc)
    return docs


def main() -> None:
    # Load goldstandard
    df = pd.read_csv(gold_path)
    df = df
    print(df[:1])
    results: List[Dict[str, Any]] = []

    # Set up reranker
    #llm = LLM(model_name=LLM.MODEL_GPT_4_1)
    #llm = LLM(model_name=LLM.MODEL_GPT_4_1_NANO)
    #llm = LLM(model_name=LLM.MODEL_GPT_4_1_MINI)
    #llm = LLM(model_name=LLM.MODEL_GEMINI_2_5_PRO)
    #llm = LLM(model_name=LLM.MODEL_GEMINI_2_5_FLASH)
    #llm = LLM(model_name=LLM.MODEL_CLAUDE_SONNET_3_7)
    llm = LLM(model_name=LLM.MODEL_CLAUDE_SONNET_4)
    reranker = LLMRerankerImage(llm, mode=LLMRerankerImage.MODE_SINGLE_SCORE)
    reranker_type = 'image'  # If you use LLMRerankerText, set to 'text'

    for counter, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df)), 1):
        query: str = str(row['query'])
        gui_indexes_str: str = str(row['gui_indexes'])
        gui_ids = ast.literal_eval(gui_indexes_str)
        # Build docs
        ranked_docs = build_ranked_docs(gui_ids)
        #print(gui_ids)
        #print(ranked_docs)
        # Rerank with timing
        start_time = time.perf_counter()
        reranked = reranker.rerank(ranked_docs, query, conf_threshold=0, top_k=100)
        #print(f"Reranked: {reranked}")
        #print(f"Reranked length: {len(reranked)}")
        #print(f"reranked: {len(ranked_docs)}")
        end_time = time.perf_counter()
        rerank_time_seconds = end_time - start_time
        # Sort by score descending
        #reranked = sorted(reranked, key=lambda r: r.score, reverse=True)
        usage = reranker.last_usage_metadata
        reranked_ids = [int(r.doc.id) for r in reranked if r.doc.id is not None]
        id_score_map = {int(r.doc.id): float(r.score) for r in reranked if r.doc.id is not None}
        results.append({
            'query': query,
            'gui_ranking': str(reranked_ids),
            'rerank_time_seconds': rerank_time_seconds,
            'input_tokens': usage.get('input_tokens', 0),
            'output_tokens': usage.get('output_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            'id_score_map': str(id_score_map),
        })
        print(f"Processed query {counter}/{len(df)} (rerank time: {rerank_time_seconds:.2f}s)")

    # Save results
    out_df = pd.DataFrame(results)
    # Build filename from LLM and reranker settings
    model_name = str(llm.model_name).replace('.', '_')
    temp = str(llm.temperature).replace('.', '_')
    out_name = f"rerank_results_{model_name}_{reranker_type}_temp{temp}.csv"
    out_path = RESULTS_DIR / out_name
    out_df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
