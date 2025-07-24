from abc import ABC, abstractmethod
from typing import Text, List


class Embeddings(ABC):
    def __init__(self, model_type: str, model_name: str):
        self.model_type = model_type
        self.model_name = model_name

    @abstractmethod
    def embed_query(self, text: Text) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def batch_embed_query(self, texts: List[Text]) -> List[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: List[Text]) -> List[List[float]]:
        raise NotImplementedError
    