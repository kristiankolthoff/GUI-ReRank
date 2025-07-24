import uuid
from abc import ABC, abstractmethod
from typing import Optional, Text, List, Dict, Any

from gui_rerank.models.ranked_ui_interface_doc import RankedUserInterfaceDocument
from gui_rerank.models.ui_interface_doc import UserInterfaceDocument
from gui_rerank.models.dataset import Dataset

class Ranker(ABC):
    def __init__(self, dataset: Dataset) -> None:
        """
        Abstract base class for all rankers. Expects a Dataset instance at initialization.
        """
        self.dataset = dataset

    @abstractmethod
    def rank(self, query: Text, conf_threshold: Optional[float] = 0.0,
             top_k: Optional[int] = 100, norm_min: Optional[int] = None,
             norm_max: Optional[int] = None) -> List[RankedUserInterfaceDocument]:
        """
        Rank the dataset (self.dataset) according to the query and return a list of ranked user interface documents.
        """
        raise NotImplementedError

    @abstractmethod
    def rank_documents(self, query: Text, documents: List[UserInterfaceDocument],
                      conf_threshold: Optional[float] = 0.0, norm_min: Optional[int] = None,
                      norm_max: Optional[int] = None) -> List[RankedUserInterfaceDocument]:
        """
        Rank the provided list of UserInterfaceDocument objects according to the query and return a list of ranked user interface documents.
        """
        raise NotImplementedError
