from gui_rerank.embeddings.lang_chain_embeddings import LangChainEmbeddings

def get_embedding_model(model_type: str, model_name: str, **kwargs):
    """
    Given a model_type and model_name, instantiate and return the correct Embeddings subclass.
    """
    if model_type == LangChainEmbeddings.MODEL_TYPE:
        return LangChainEmbeddings(model_name=model_name, **kwargs)
    raise ValueError(f"Unknown embedding model type: {model_type}")
