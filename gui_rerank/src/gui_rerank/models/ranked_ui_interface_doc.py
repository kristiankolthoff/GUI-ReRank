from typing import Optional, Dict, Text, Any
from .ui_interface_doc import UserInterfaceDocument

class RankedUserInterfaceDocument:

    def __init__(self, doc: UserInterfaceDocument, score: float, rank: int, source: Optional[str] = "", dimension_scores: Optional[Dict[str, float]] = None):
        self.doc = doc
        self.score = score
        self.rank = rank
        self.source = source
        self.dimension_scores = dimension_scores or {}

    def to_dict(self) -> Dict[Text, Any]:
        return {
            'doc': self.doc.to_dict() if self.doc else None,
            'score': self.score,
            'rank': self.rank,
            'source': self.source,
            'dimension_scores': self.dimension_scores
        }

    @classmethod
    def from_dict(cls, d: Dict[Text, Any]) -> 'RankedUserInterfaceDocument':
        from .ui_interface_doc import UserInterfaceDocument
        if d.get('doc'):
            doc = UserInterfaceDocument.from_dict(d['doc'])
        else:
            doc = UserInterfaceDocument(filepath='', text=None, annotation=None)
        return cls(
            doc=doc,
            score=d.get('score', 0),
            rank=d.get('rank', 0),
            source=d.get('source', ''),
            dimension_scores=d.get('dimension_scores', {})
        )

    def __eq__(self, other):
        if not isinstance(other, RankedUserInterfaceDocument):
            return False
        return (self.doc == other.doc and self.score == other.score and self.rank == other.rank and self.source == other.source and self.dimension_scores == other.dimension_scores)

    def __hash__(self):
        return hash((self.doc, self.score, self.rank, self.source, tuple(sorted(self.dimension_scores.items()))))

    def __str__(self):
        return f"RankedUserInterfaceDocument(doc={self.doc}, score={self.score}, rank={self.rank}, source={self.source}, dimension_scores={self.dimension_scores})"

    def __repr__(self):
        return (
            f"RankedUserInterfaceDocument(doc={repr(self.doc)}, score={self.score}, "
            f"rank={self.rank}, source={repr(self.source)}, dimension_scores={repr(self.dimension_scores)})"
        )
