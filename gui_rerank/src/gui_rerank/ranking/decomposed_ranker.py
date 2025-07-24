from gui_rerank.models.search_dimension import SearchDimension
import numpy as np
from typing import Optional, Text, List, Dict
from sklearn.metrics.pairwise import cosine_similarity

from gui_rerank.models.ranked_ui_interface_doc import RankedUserInterfaceDocument
from gui_rerank.models.ui_interface_doc import UserInterfaceDocument
from gui_rerank.models.dataset import Dataset
from gui_rerank.embeddings.embeddings import Embeddings
from gui_rerank.ranking.ranker import Ranker
from gui_rerank.query_decomposition.query_decomposition import QueryDecomposer
from gui_rerank.annotator.config import DEFAULT_SEARCH_DIMENSIONS

class DecomposedRanker(Ranker):
    def __init__(self, dataset: Dataset, embedding_model: Embeddings,
                 search_dimensions: Optional[List[SearchDimension]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 pos_neg_weights: Optional[Dict[str, Dict[str, float]]] = None):
        super().__init__(dataset)
        self.embedding_model = embedding_model
        self.search_dimensions = search_dimensions if search_dimensions is not None else DEFAULT_SEARCH_DIMENSIONS
        self.decomposer = QueryDecomposer(search_dimensions=self.search_dimensions)
        # Use defaults from search_dimensions if not provided
        if weights is None:
            weights = {d.name: (d.weight if d.weight is not None else 0.5) for d in self.search_dimensions if d.query_decomposition}
        if pos_neg_weights is None:
            pos_neg_weights = {d.name: {'pos': (d.pos_weight if d.pos_weight is not None else 1.0), 'neg': (d.neg_weight if d.neg_weight is not None else 1.0)} for d in self.search_dimensions if d.negation}
        self.weights = weights
        print(f"Weights from decomposed_ranker: {self.weights}")
        self.pos_neg_weights = pos_neg_weights

    def rank(self, query: Text, conf_threshold: Optional[float] = 0.0,
             top_k: Optional[int] = 100, norm_min: Optional[int] = None,
             norm_max: Optional[int] = None) -> List[RankedUserInterfaceDocument]:
        # Decompose the query
        parsed_query = self.decomposer.decompose_query(query)
        print(f"Parsed query: {parsed_query}")
        num_items = next(iter(self.dataset.embeddings.values())).shape[0]
        print(f"Number of items in dataset: {num_items}")
        final_scores = np.zeros(num_items)
        total_weight = 0.0
        # Store per-dimension weighted scores for each doc
        per_dimension_scores = {aspect: np.zeros(num_items) for aspect in [d.name for d in self.search_dimensions if d.query_decomposition]}

        for dim in self.search_dimensions:
            if not dim.query_decomposition:
                continue
            aspect = dim.name
            if aspect not in self.dataset.embeddings:
                print(f"Skipping aspect '{aspect}' - not found in dataset embeddings")
                continue
            
            print(f"\nProcessing aspect: {aspect}")
            matrix = self.dataset.embeddings[aspect]
            print(f"Embedding matrix shape for {aspect}: {matrix.shape}")
            
            # Always expect {aspect: {"pos": [...], "neg": [...]}}
            val = parsed_query.get(aspect, {})
            if isinstance(val, dict):
                pos_phrases = [p for p in val.get('pos', []) if p.strip()]
                neg_phrases = [p for p in val.get('neg', []) if p.strip()]
            else:
                pos_phrases = []
                neg_phrases = []
            
            print(f"Positive phrases for {aspect}: {pos_phrases}")
            print(f"Negative phrases for {aspect}: {neg_phrases}")

            pos_scores = np.zeros(num_items)
            neg_scores = np.zeros(num_items)

            # Positive phrases
            for phrase in pos_phrases:
                emb = np.array(self.embedding_model.embed_query(phrase)).reshape(1, -1)
                phrase_scores = cosine_similarity(emb, matrix)[0]
                pos_scores += phrase_scores
                print(f"  Positive phrase '{phrase}' - min score: {phrase_scores.min():.4f}, max score: {phrase_scores.max():.4f}, mean: {phrase_scores.mean():.4f}")
            
            # Negative phrases
            for phrase in neg_phrases:
                emb = np.array(self.embedding_model.embed_query(phrase)).reshape(1, -1)
                phrase_scores = cosine_similarity(emb, matrix)[0]
                neg_scores += phrase_scores
                print(f"  Negative phrase '{phrase}' - min score: {phrase_scores.min():.4f}, max score: {phrase_scores.max():.4f}, mean: {phrase_scores.mean():.4f}")
            
            # Normalize
            if pos_phrases:
                pos_scores /= len(pos_phrases)
                print(f"  Normalized positive scores - min: {pos_scores.min():.4f}, max: {pos_scores.max():.4f}, mean: {pos_scores.mean():.4f}")
            if neg_phrases:
                neg_scores /= len(neg_phrases)
                print(f"  Normalized negative scores - min: {neg_scores.min():.4f}, max: {neg_scores.max():.4f}, mean: {neg_scores.mean():.4f}")
            
            # Apply weights
            aspect_w = self.weights.get(aspect, 1.0)
            pos_w = self.pos_neg_weights.get(aspect, {}).get('pos', 1.0)
            neg_w = self.pos_neg_weights.get(aspect, {}).get('neg', 1.0)
            print(f"  Weights - aspect: {aspect_w}, pos: {pos_w}, neg: {neg_w}")
            
            if (pos_w + neg_w) > 0:
                weighted_score = ((pos_scores * pos_w - neg_scores * neg_w) / (pos_w + neg_w)) * aspect_w
                print(f"  Weighted score - min: {weighted_score.min():.4f}, max: {weighted_score.max():.4f}, mean: {weighted_score.mean():.4f}")
            else:
                weighted_score = np.zeros(num_items)
                print(f"  Weighted score - zero (no pos/neg weights)")
            
            final_scores += weighted_score
            total_weight += aspect_w
            print(f"  Running total weight: {total_weight}")
            # Store per-dimension weighted score
            per_dimension_scores[aspect] = weighted_score.copy()

        # Normalize final score by total weight
        if total_weight > 0:
            final_scores /= total_weight
            for aspect in per_dimension_scores:
                per_dimension_scores[aspect] /= total_weight
            print(f"\nFinal scores after normalization - min: {final_scores.min():.4f}, max: {final_scores.max():.4f}, mean: {final_scores.mean():.4f}")
        else:
            print(f"\nWarning: total_weight is {total_weight}, skipping normalization")
        
        sorted_indices = np.argsort(final_scores)[::-1]
        print(f"Top 10 scores: {final_scores[sorted_indices[:10]]}")
        
        results = []
        for rank, idx in enumerate(sorted_indices[:top_k]):
            score = float(final_scores[idx])
            if conf_threshold is not None and score < conf_threshold:
                continue
            image_id = self.dataset.index_mapping[idx]
            doc = UserInterfaceDocument(id=image_id)
            # Gather per-dimension scores for this doc
            doc_dimension_scores = {aspect: float(per_dimension_scores[aspect][idx]) for aspect in per_dimension_scores}
            results.append(RankedUserInterfaceDocument(doc=doc, score=score, rank=rank+1, source="decomposed_ranker", dimension_scores=doc_dimension_scores))
        
        print(f"\nReturning {len(results)} results (top_k={top_k}, conf_threshold={conf_threshold})")
        return results

    def rank_documents(self, query: Text, documents: List[UserInterfaceDocument],
                        conf_threshold: Optional[float] = 0.0, norm_min: Optional[int] = None,
                        norm_max: Optional[int] = None) -> List[RankedUserInterfaceDocument]:
            raise NotImplementedError("DecomposedRanker does not support ranking arbitrary document lists.")

if __name__ == "__main__":
    import os
    from gui_rerank.embeddings.lang_chain_embeddings import LangChainEmbeddings

    # Path setup (adjust as needed)
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/datasets/'))
    dataset_name = "demo"  # Should match the name used in dataset_builder.py

    # Load dataset
    dataset = Dataset.load(dataset_dir, name=dataset_name)
    print(f"Loaded dataset: {dataset}")

    # Create embedding model (should match what was used in dataset_builder.py)
    embedding_model = LangChainEmbeddings(model_name=LangChainEmbeddings.OPENAI_TEXT_EMBEDDING_LARGE_3, dimensions=3072)

    # Instantiate DecomposedRanker
    ranker = DecomposedRanker(dataset, embedding_model)

    # Run a sample query
    query = "A recipe app showing a list of recipes"
    results = ranker.rank(query, top_k=5)

    print("\nTop 5 results:")
    for r in results:
        print(f"Rank: {r.rank}, Score: {r.score:.4f}, ID: {r.doc.id}")