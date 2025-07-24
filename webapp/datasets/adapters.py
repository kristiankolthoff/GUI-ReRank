import os
from django.conf import settings
from gui_rerank.models.screen_image import ScreenImage

from .media_config import DATASETS_BASE_PATH
from .models import ImageEntry, Dataset
from typing import Any, Dict
from gui_rerank.models.dataset import Dataset
from gui_rerank.models.search_dimension import SearchDimension as SearchDimensionDC

def imageentry_to_screenimage(image_entry: ImageEntry) -> ScreenImage:
    """
    Convert a Django ImageEntry instance to a ScreenImage instance.
    """
    image_id = str(getattr(image_entry, 'id', getattr(image_entry, 'pk', '')))
    file_field = getattr(image_entry, 'file', None)
    file_path = getattr(file_field, 'path', None)
    file_url = getattr(file_field, 'url', None)
    filelocation = file_path if file_path else file_url if file_url else ''
    return ScreenImage(
        id=image_id,
        filelocation=filelocation
    )

def screenimage_to_imageentry_dict(screen_image: ScreenImage, dataset: Dataset, annotation: str = "") -> Dict[str, Any]:
    """
    Prepare a dict for creating an ImageEntry from a ScreenImage.
    (Does not save the file, just prepares the fields.)
    dataset: a Dataset model instance
    annotation: annotation string (JSON or plain text)
    """
    return {
        'dataset': dataset,
        'name': screen_image.id,
        'file': screen_image.filelocation,  # This should be a File or path, may need adaptation
        'annotation': annotation,
    }

def django_dataset_to_framework_dataset(django_dataset):
    """
    Given a Django Dataset model instance, load and return the framework Dataset object.
    """
    dataset_path = os.path.join(settings.MEDIA_ROOT, DATASETS_BASE_PATH)
    embedding_model_type = django_dataset.embedding_model_type
    embedding_model_name = django_dataset.embedding_model_name
    print(f"Dataset path: {dataset_path}")
    return Dataset.load(dataset_path, str(django_dataset.id), embedding_model_type, embedding_model_name)

def searchdimension_from_model(model_instance):
    return SearchDimensionDC(
        name=model_instance.name,
        annotation_description=model_instance.annotation_description,
        type=model_instance.type,
        query_decomposition=model_instance.query_decomposition,
        negation=model_instance.negation,
        weight=model_instance.weight,
        pos_weight=model_instance.pos_weight,
        neg_weight=model_instance.neg_weight
    )
