from typing import List, Text, Optional

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from gui_rerank.embeddings.embeddings import Embeddings
from langchain_core.embeddings.embeddings import Embeddings as EmbeddingModel

load_dotenv()


class LangChainEmbeddings(Embeddings):

    MODEL_TYPE = "LangChainEmbeddings"
    # Defines the available variants of this ranking model class
    OPENAI_TEXT_EMBEDDING_LARGE_3 = 'openai_text_embedding_large_3'
    GEMINI_TEXT_EMBEDDING_0307 = 'gemini_embedding_exp_03_07'

    def __init__(self, model_name: Optional[Text] = OPENAI_TEXT_EMBEDDING_LARGE_3,
                 dimensions: Optional[int] = None):
        super().__init__(model_type=LangChainEmbeddings.MODEL_TYPE, model_name=model_name)
        self.dimensions = dimensions
        self.embedding_model = LangChainEmbeddings._get_embedding_model(model_name=model_name, dimensions=dimensions)

    def embed_query(self, text: Text) -> List[float]:
        return self.embedding_model.embed_query(text=text)

    def batch_embed_query(self, texts: List[Text]) -> List[List[float]]:
        return self.embed_documents(texts=texts)

    def embed_documents(self, texts: List[Text]) -> List[List[float]]:
        return self.embedding_model.embed_documents(texts=texts)

    @staticmethod
    def _get_embedding_model(model_name: Text, dimensions: Optional[int] = None) -> EmbeddingModel:
        """Load the corresponding embedding model given the model name and config"""
        if model_name == LangChainEmbeddings.OPENAI_TEXT_EMBEDDING_LARGE_3:
            if not dimensions:
                return OpenAIEmbeddings(model='text-embedding-3-large', dimensions=3072)
            return OpenAIEmbeddings(model='text-embedding-3-large', dimensions=dimensions)
        elif model_name == LangChainEmbeddings.GEMINI_TEXT_EMBEDDING_0307:
            return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")


if __name__ == "__main__":
    embeddings = LangChainEmbeddings(model_name=LangChainEmbeddings.OPENAI_TEXT_EMBEDDING_LARGE_3, dimensions=3072)
    embedded_query = embeddings.embed_query("this is a test query")
    print(embedded_query)
    embedded_docs = embeddings.embed_documents(["this is a second test", "this is a test query"])
    print(embedded_docs)
    print(len(embedded_query))
    print(len(embedded_docs))