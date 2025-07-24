from typing import List
from gui_rerank.models.search_dimension import SearchDimension

# Template for the general_description annotation description
GENERAL_DESCRIPTION_TEMPLATE = (
    "Describe the entire GUI screenshot with its {dimensions} comprehensively."
)

# Create all search dimensions except general_description
DEFAULT_SEARCH_DIMENSIONS = [
    SearchDimension(
        name="domain",
        annotation_description="Describe the domain of the GUI screenshot.",
        type="embedding",
        query_decomposition=True,
        negation=True,
        weight=0.5,
        pos_weight=5.0,
        neg_weight=1.0
    ),
    SearchDimension(
        name="functionality",
        annotation_description="Describe the main functionality of the GUI screenshot comprehensively.",
        type="embedding",
        query_decomposition=True,
        negation=True,
        weight=0.8,
        pos_weight=5.0,
        neg_weight=1.0
    ),
    SearchDimension(
        name="design",
        annotation_description="Describe the design of the GUI screenshot comprehensively.",
        type="embedding",
        query_decomposition=True,
        negation=True,
        weight=0.5,
        pos_weight=5.0,
        neg_weight=1.0
    ),
    SearchDimension(
        name="gui_components",
        annotation_description="Describe the GUI components of the GUI screenshot comprehensively. Name the component type and additional information",
        type="embedding",
        query_decomposition=True,
        negation=True,
        weight=0.5,
        pos_weight=5.0,
        neg_weight=1.0
    ),
    SearchDimension(
        name="text",
        annotation_description="List of all texts displayed on the screen as a single string.",
        type="embedding",
        query_decomposition=True,
        negation=True,
        weight=0.5,
        pos_weight=5.0,
        neg_weight=1.0
    ),
]

def create_general_description_dimension(search_dimensions: List[SearchDimension]) -> SearchDimension:
    other_dimension_names = ", ".join([d.name for d in search_dimensions])
    return SearchDimension(
        name="general_description",
        annotation_description=GENERAL_DESCRIPTION_TEMPLATE.format(dimensions=other_dimension_names),
        type="embedding",
        query_decomposition=False,
        negation=False,
        weight=1.0,
        pos_weight=1.0,
        neg_weight=1.0
    )

DEFAULT_SEARCH_DIMENSIONS.append(create_general_description_dimension(DEFAULT_SEARCH_DIMENSIONS))

# Create the list with NSFW dimension by copying the default list
DEFAULT_SEARCH_DIMENSIONS_WITH_NSFW = DEFAULT_SEARCH_DIMENSIONS.copy()
DEFAULT_SEARCH_DIMENSIONS_WITH_NSFW.append(
    SearchDimension(
        name="nsfw",
        annotation_description="Provide a boolean value True or False if the screenshot is NSFW.",
        type=None,
        query_decomposition=False,
        negation=False,
        weight=1.0,
        pos_weight=1.0,
        neg_weight=1.0
    )
)

SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png']
