import numpy as np
from typing import Optional, Text, List
from sklearn.metrics.pairwise import cosine_similarity

from gui_rerank.models.ranked_ui_interface_doc import RankedUserInterfaceDocument
from gui_rerank.models.ui_interface_doc import UserInterfaceDocument
from gui_rerank.models.dataset import Dataset
from gui_rerank.embeddings.embeddings import Embeddings
from gui_rerank.ranking.ranker import Ranker

class SimpleRanker(Ranker):
    def __init__(self, dataset: Dataset, embedding_model: Embeddings) -> None:
        super().__init__(dataset)
        self.embedding_model = embedding_model

    def rank(self, query: Text, conf_threshold: Optional[float] = 0.0,
             top_k: Optional[int] = 100, norm_min: Optional[int] = None,
             norm_max: Optional[int] = None) -> List[RankedUserInterfaceDocument]:
        # Embed the query
        query_emb = np.array(self.embedding_model.embed_query(query)).reshape(1, -1)
        doc_embs = self.dataset.embeddings["general_description"]
        # Compute cosine similarity using sklearn
        scores = cosine_similarity(query_emb, doc_embs)[0]
        sorted_indices = np.argsort(scores)[::-1]
        # Prepare results
        results = []
        for rank, idx in enumerate(sorted_indices[:top_k]):
            score = scores[idx]
            if conf_threshold is not None and score < conf_threshold:
                continue
            # Map index to image_id
            image_id = self.dataset.index_mapping[idx]
            annotation = self.dataset.annotations[str(image_id)]
            # Create a UserInterfaceDocument (minimal, as we only have id)
            doc = UserInterfaceDocument(id=image_id, annotation=annotation)
            results.append(RankedUserInterfaceDocument(doc=doc, score=score, rank=rank+1, source="simple_ranker"))
        return results

    def rank_documents(self, query: Text, documents: List[UserInterfaceDocument],
                      conf_threshold: Optional[float] = 0.0, norm_min: Optional[int] = None,
                      norm_max: Optional[int] = None) -> List[RankedUserInterfaceDocument]:
        raise NotImplementedError("SimpleRanker does not support ranking arbitrary document lists.")

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

    # Instantiate SimpleRanker
    ranker = SimpleRanker(dataset, embedding_model)

    # Run a sample query
    #query = "A music player with black interface"
    query = "A recipe app showing a list of recipes"
    results = ranker.rank(query, top_k=5)

    print("\nTop 5 results:")
    for r in results:
        print(f"Rank: {r.rank}, Score: {r.score:.4f}, ID: {r.doc.filepath}")
