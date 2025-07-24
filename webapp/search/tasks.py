import json
import os
from typing import List
from celery import shared_task
from gui_rerank.llm.llm import LLM
from gui_rerank.reranking.llm_reranker_text import LLMRerankerText
from gui_rerank.models.ranked_ui_interface_doc import RankedUserInterfaceDocument
from gui_rerank.reranking.llm_reranker_image import LLMRerankerImage

from django.conf import settings

from datasets.models import Dataset as DjangoDataset, SearchDimension as DjangoSearchDimension, ImageEntry
from settings.models import set_api_keys_from_settings

@shared_task
def rerank_gui_batch(batch_data: List[dict], query: str, llm_reranker_params_json: str):
    set_api_keys_from_settings()
    print(f"Reranking batch of {len(batch_data)} documents")
    # Parse LLM reranker params
    llm_reranker_params = json.loads(llm_reranker_params_json)
    llm_model = llm_reranker_params.get('llm_model', LLM.MODEL_GPT_4_1)
    llm_type = llm_reranker_params.get('llm_type', 'text')
    llm_temp = llm_reranker_params.get('llm_temp', 0.05)
    reranker_mode = llm_reranker_params.get('reranker_mode', LLMRerankerText.MODE_SINGLE_SCORE)
    reranker_weights = llm_reranker_params.get('reranker_weights', None)
    dataset_id = llm_reranker_params.get('dataset_id', None)
    llm_weights = llm_reranker_params.get('llm_weights', None)

    # Reconstruct search dimensions for this dataset and update weights
    search_dimensions = None
    if dataset_id is not None:
        from datasets.adapters import searchdimension_from_model
        try:
            django_dataset = DjangoDataset.objects.get(id=dataset_id)  # type: ignore[attr-defined]
            dims = DjangoSearchDimension.objects.filter(dataset=django_dataset)  # type: ignore[attr-defined]
            search_dimensions = [searchdimension_from_model(d) for d in dims if d.query_decomposition]
        except Exception as e:
            print(f"Error fetching search dimensions for reranking: {e}")
            from gui_rerank.annotator.config import DEFAULT_SEARCH_DIMENSIONS
            search_dimensions = [d for d in DEFAULT_SEARCH_DIMENSIONS if d.query_decomposition]
        # Update weights from llm_weights if provided
        if llm_weights:
            for dim in search_dimensions:
                if dim.name in llm_weights:
                    dim.weight = llm_weights[dim.name]

    # Instantiate LLM and reranker
    llm = LLM(model_name=llm_model, temperature=llm_temp)
    if llm_type == 'image':
        reranker = LLMRerankerImage(llm, mode=reranker_mode, 
                        weights=reranker_weights, search_dimensions=search_dimensions)
    else:
        reranker = LLMRerankerText(llm, mode=reranker_mode, 
                        weights=reranker_weights, search_dimensions=search_dimensions)
    # Deserialize batch_data into RankedUserInterfaceDocument objects
    ranked_docs = [RankedUserInterfaceDocument.from_dict(d) for d in batch_data]
    for ranked_doc in ranked_docs:
        try:
            image_entry = ImageEntry.objects.get(id=ranked_doc.doc.id)  # type: ignore[attr-defined]
            image_url = image_entry.file.path if image_entry.file else ''
            annotation = json.loads(image_entry.annotation)
            ranked_doc.doc.annotation = annotation
            ranked_doc.doc.filepath = image_entry.file.path
            print(f"Image URL: {image_url}")
            print(f"Image entry file path: {image_entry.file.path}")
            print(f"Image entry file: {image_entry.file}")
            print(f"Image entry file url: {image_entry.file.url}")
            print(f"Joined path: {os.path.join(image_entry.file.path, image_entry.file.url)}")
        except ImageEntry.DoesNotExist:  # type: ignore[attr-defined]
            image_url = ''
    print(f"Ranked docs: {ranked_docs}")
    results_reranked = reranker.rerank(ranked_docs, query)
    print(f"Reranked docs: {results_reranked}")
    return [r.to_dict() for r in results_reranked]
