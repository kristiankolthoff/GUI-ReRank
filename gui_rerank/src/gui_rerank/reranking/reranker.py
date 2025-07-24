from abc import ABC, abstractmethod
from typing import List, Optional
from gui_rerank.models.ranked_ui_interface_doc import RankedUserInterfaceDocument
from gui_rerank.models.ui_interface_doc import UserInterfaceDocument

class Reranker(ABC):
    """
    Abstract base class for rerankers that take a list of already ranked user interfaces and rerank them.
    """
    def __init__(self):
        pass

    @abstractmethod
    def rerank(self, ranked_docs: List[RankedUserInterfaceDocument], query: str, 
               conf_threshold: Optional[float] = 0.0,
               top_k: Optional[int] = 100, norm_min: Optional[int] = None,
               norm_max: Optional[int] = None) -> List[RankedUserInterfaceDocument]:
        """
        Rerank a list of RankedUserInterfaceDocument objects given a query and return a new list.
        """
        raise NotImplementedError("Subclasses must implement this method")