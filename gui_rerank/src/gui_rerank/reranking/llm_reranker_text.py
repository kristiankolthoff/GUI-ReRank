import os
import json
from pathlib import Path
from typing import List, Optional
from gui_rerank.llm.llm import LLM
from gui_rerank.models.ui_interface_doc import UserInterfaceDocument
from gui_rerank.reranking.reranker import Reranker
from gui_rerank.models.ranked_ui_interface_doc import RankedUserInterfaceDocument
from gui_rerank.annotator.config import DEFAULT_SEARCH_DIMENSIONS
from gui_rerank.models.search_dimension import SearchDimension

class LLMRerankerText(Reranker):

    PLACEHOLDER_QUERY = "{query}"
    PLACEHOLDER_DOCS = "{docs}"
    PLACEHOLDER_RELEVANT_INFO = "{placeholder_relevant_info}"
    PLACEHOLDER_RATING_DESCRIPTION = "{placeholder_rating_description}"

    PROMPT_TEMPLATE_PATH_SINGLE_SCORE = "resources/prompts/llm_rerank_text.txt"
    PROMPT_TEMPLATE_PATH_DECOMPOSED = "resources/prompts/llm_rerank_text_decomposed.txt"

    MODE_SINGLE_SCORE = "single_score"
    MODE_DECOMPOSED = "decomposed"

    def __init__(self, llm: LLM, mode: str = MODE_SINGLE_SCORE, prompt_template_path: Optional[str] = None,
                 batch_size: int = 5, weights: Optional[dict] = None, search_dimensions: Optional[List[SearchDimension]] = None):
        super().__init__()
        self.llm = llm
        self.batch_size = batch_size
        self.mode = mode
        self.search_dimensions = search_dimensions if search_dimensions is not None else DEFAULT_SEARCH_DIMENSIONS
        if weights is None:
            weights = {d.name: (d.weight if d.weight is not None else 1.0) for d in self.search_dimensions if d.query_decomposition}
        self.weights = weights
        print(f"Weights LLM reranker text: {self.weights}")
        if self.mode == self.MODE_DECOMPOSED:
            if prompt_template_path is None:
                prompt_template_path = self.PROMPT_TEMPLATE_PATH_DECOMPOSED
        else:
            if prompt_template_path is None:
                prompt_template_path = self.PROMPT_TEMPLATE_PATH_SINGLE_SCORE
        prompt_template = LLMRerankerText._load_prompt_template(prompt_template_path)
        if self.mode == self.MODE_DECOMPOSED:
            relevant_info = self._extract_relevant_info(self.search_dimensions)
            rating_description = self._extract_rating_description(self.search_dimensions)
            prompt_template = prompt_template.replace(self.PLACEHOLDER_RELEVANT_INFO, relevant_info)
            prompt_template = prompt_template.replace(self.PLACEHOLDER_RATING_DESCRIPTION, rating_description)
        self.prompt_template = prompt_template
        self.last_usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    @staticmethod
    def _load_prompt_template(prompt_template_path: str) -> str:
        """
        Load the prompt template from file.
        
        Returns:
            The prompt template as a string
            
        Raises:
            FileNotFoundError: If the prompt template file doesn't exist
        """
        # Try to find the prompt template file
        template_paths = [
            Path(prompt_template_path),
            Path("gui_rerank") / prompt_template_path,
            Path(__file__).parent.parent / prompt_template_path,
            Path(__file__).parent.parent.parent / prompt_template_path,
            Path(__file__).parent.parent.parent.parent / prompt_template_path,
            Path(__file__).parent.parent.parent.parent.parent / prompt_template_path,
        ]
        
        for path in template_paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
        
        raise FileNotFoundError(f"Prompt template not found. Tried paths: {template_paths}")

    @staticmethod
    def _extract_relevant_info(search_dimensions: List[SearchDimension]) -> str:
        return ", ".join([d.name for d in search_dimensions if d.query_decomposition])

    @staticmethod
    def _extract_rating_description(search_dimensions: List[SearchDimension]) -> str:
        return json.dumps({d.name: d.rating_description or '' for d in search_dimensions if d.query_decomposition})

    def rerank(self, ranked_docs: List[RankedUserInterfaceDocument], query: str,
              conf_threshold: Optional[float] = 0.0,
              top_k: Optional[int] = 100, norm_min: Optional[int] = None,
              norm_max: Optional[int] = None) -> List[RankedUserInterfaceDocument]:
        scored_docs = []
        reranked = []
        usage_acc = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for batch_start in range(0, len(ranked_docs), self.batch_size):
            batch = ranked_docs[batch_start:self.batch_size+batch_start]
            doc_id_map = {}
            doc_descriptions = []
            for idx, ranked_doc in enumerate(batch):
                doc_id = str(idx)
                doc_id_map[doc_id] = ranked_doc
                annotation = None
                if hasattr(ranked_doc.doc, 'annotation'):
                    annotation = ranked_doc.doc.annotation
                if annotation and isinstance(annotation, str):
                    try:
                        annotation = json.loads(annotation)
                    except Exception:
                        pass
                description = annotation.get('general_description', '') if annotation else ''
                doc_descriptions.append(f"{doc_id}: {description}")
            docs_str = "\n".join(doc_descriptions)
            prompt = self.prompt_template.replace(self.PLACEHOLDER_QUERY, query).replace(self.PLACEHOLDER_DOCS, docs_str)
            print(f"Prompt: {prompt}")
            llm_response, usage_metadata = self.llm.invoke(prompt)
            # Accumulate usage
            for k in ("input_tokens", "output_tokens", "total_tokens"):
                if isinstance(usage_metadata, dict):
                    if k in usage_metadata:
                        usage_acc[k] += int(usage_metadata[k])
            try:
                scores = json.loads(llm_response.replace("```json", "").replace("```", ""))
                print(f"Scores: {scores}")
            except Exception:
                scores = {}
            if self.mode == self.MODE_SINGLE_SCORE:
                for doc_id, ranked_doc in doc_id_map.items():
                    score = scores.get(doc_id, 0)
                    norm_score = float(score) / 100.0
                    scored_docs.append((norm_score, ranked_doc))
            elif self.mode == self.MODE_DECOMPOSED:
                for doc_id, ranked_doc in doc_id_map.items():
                    aspect_scores = scores.get(doc_id, {})
                    weighted_sum = 0.0
                    total_weight = 0.0
                    for aspect, weight in self.weights.items():
                        aspect_score = aspect_scores.get(aspect, 0)
                        weighted_sum += float(aspect_score) * weight
                        total_weight += weight
                    norm_score = (weighted_sum / total_weight) / 100.0 if total_weight > 0 else 0.0
                    norm_aspect_scores = {aspect: (float(aspect_scores.get(aspect, 0)) / 100.0) for aspect in self.weights}
                    scored_docs.append((norm_score, ranked_doc, norm_aspect_scores))
        if self.mode == self.MODE_DECOMPOSED:
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                for new_rank, (norm_score, ranked_doc, norm_aspect_scores) in enumerate(scored_docs, 1):
                    reranked.append(RankedUserInterfaceDocument(
                        doc=ranked_doc.doc,
                        score=norm_score,
                        rank=new_rank,
                        source='llm_rerank_text',
                        dimension_scores=norm_aspect_scores
                    ))
        else:
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                for new_rank, (norm_score, ranked_doc) in enumerate(scored_docs, 1):
                    reranked.append(RankedUserInterfaceDocument(
                        doc=ranked_doc.doc,
                        score=norm_score,
                        rank=new_rank,
                        source='llm_rerank_text'
                    ))
        self.last_usage_metadata = usage_acc
        return reranked


if __name__ == "__main__":
    import sys
    from gui_rerank.models.dataset import Dataset
    from gui_rerank.embeddings.lang_chain_embeddings import LangChainEmbeddings
    from gui_rerank.ranking.simple_ranker import SimpleRanker

    # Paths
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/datasets/'))
    dataset_name = "demo"

    # Load dataset
    dataset = Dataset.load(dataset_dir, name=dataset_name)
    print(f"Loaded dataset: {dataset}")

    # Create embedding model (should match what was used in dataset_builder.py)
    embedding_model = LangChainEmbeddings(model_name=LangChainEmbeddings.OPENAI_TEXT_EMBEDDING_LARGE_3, dimensions=3072)

    # Instantiate SimpleRanker
    ranker = SimpleRanker(dataset, embedding_model)

    # Instantiate LLM
    llm = LLM()

    # Instantiate LLMRerankerText
    reranker = LLMRerankerText(llm, mode=LLMRerankerText.MODE_DECOMPOSED, 
                               search_dimensions=DEFAULT_SEARCH_DIMENSIONS)

    # Run a sample query
    query = "A recipe app showing a list of recipes"
    results = ranker.rank(query, top_k=10)

    print("\nTop 10 ranked results:")
    for r in results:
        print(r)

    # Attach annotation to each doc for reranker
    #for r in results:
     #   img_id = r.doc.filepath
     #   r.doc.annotation = dataset.annotations.get(img_id, {})

    reranked = reranker.rerank(results, query, top_k=5)

    print("\nTop 10 reranked results:")
    for r in reranked:
        print(r)