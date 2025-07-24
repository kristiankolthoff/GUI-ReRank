from gui_rerank.embeddings.lang_chain_embeddings import LangChainEmbeddings

AVAILABLE_EMBEDDING_MODELS = [
    (LangChainEmbeddings.MODEL_TYPE, LangChainEmbeddings.OPENAI_TEXT_EMBEDDING_LARGE_3, 'OpenAI Text Embedding 3 Large'),
] 