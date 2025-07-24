from dataclasses import dataclass
from typing import Optional

@dataclass
class SearchDimension:
    name: str
    annotation_description: str
    type: Optional[str] = "embedding"
    query_decomposition: Optional[bool] = True
    negation: Optional[bool] = True
    weight: Optional[float] = 0.5
    pos_weight: Optional[float] = 5.0
    neg_weight: Optional[float] = 1.0
    rating_description: Optional[str] = None

    def __post_init__(self):
        if self.rating_description is None:
            # Preprocess name: replace underscores with spaces and capitalize each word
            pretty_name = self.name.replace('_', ' ')
            self.rating_description = f"Rate the {pretty_name} of the GUI screenshot given the query."

    def __str__(self):
        return (f"SearchDimension(name={self.name!r}, annotation_description={self.annotation_description!r}, "
                f"type={self.type!r}, query_decomposition={self.query_decomposition}, negation={self.negation}, "
                f"weight={self.weight}, pos_weight={self.pos_weight}, neg_weight={self.neg_weight}, rating_description={self.rating_description!r})")

    def __repr__(self):
        return (f"SearchDimension(name={self.name!r}, annotation_description={self.annotation_description!r}, "
                f"type={self.type!r}, query_decomposition={self.query_decomposition}, negation={self.negation}, "
                f"weight={self.weight}, pos_weight={self.pos_weight}, neg_weight={self.neg_weight}, rating_description={self.rating_description!r})")