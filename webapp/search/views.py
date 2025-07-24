from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from gui_rerank.embeddings.utils import get_embedding_model
from gui_rerank.models.ranked_ui_interface_doc import RankedUserInterfaceDocument
from datasets.models import Dataset as DjangoDataset, SearchDimension as DjangoSearchDimension, ImageEntry
from datasets.adapters import django_dataset_to_framework_dataset, searchdimension_from_model
from django.core.cache import cache
from gui_rerank.ranking.simple_ranker import SimpleRanker
from datasets.models import ImageEntry
from django.conf import settings
import json
import os
from typing import List
from celery import group

from .config import BATCH_SIZE_CELERY_SEARCH, CELERY_SEARCH_TIMEOUT
from .tasks import rerank_gui_batch
from gui_rerank.ranking.decomposed_ranker import DecomposedRanker
from gui_rerank.reranking.llm_reranker_text import LLMRerankerText
from gui_rerank.reranking.llm_reranker_image import LLMRerankerImage
from gui_rerank.llm.llm import LLM
from gui_rerank.annotator.config import DEFAULT_SEARCH_DIMENSIONS
from gui_rerank.llm.config import AVAILABLE_MODELS
from settings.models import set_api_keys_from_settings

os.environ.pop("SSL_CERT_FILE", None)

def search_view(request):
    print(f"Search view called")
    # Try to get the active dataset id from the session
    dataset_id = request.session.get('active_dataset_id')
    if dataset_id:
        try:
            django_dataset = DjangoDataset.objects.get(id=dataset_id)  # type: ignore[attr-defined]
            dims = DjangoSearchDimension.objects.filter(dataset=django_dataset)  # type: ignore[attr-defined]
            dimensions = [
                {'name': d.name, 'weight': d.weight}
                for d in dims if d.query_decomposition
            ]
        except DjangoDataset.DoesNotExist:  # type: ignore[attr-defined]
            dimensions = [
                {'name': d.name, 'weight': d.weight}
                for d in DEFAULT_SEARCH_DIMENSIONS if d.query_decomposition
            ]
    else:
        dimensions = [
            {'name': d.name, 'weight': d.weight}
            for d in DEFAULT_SEARCH_DIMENSIONS if d.query_decomposition
        ]
    llm_options = AVAILABLE_MODELS
    default_llm = LLM.MODEL_GPT_4_1
    return render(request, 'search/search.html', {'search_dimensions': dimensions, 'llm_options': llm_options, 'default_llm': default_llm})

def get_active_framework_dataset(request):
    cache_key = f"active_framework_dataset_{request.session.session_key}"
    dataset = cache.get(cache_key)
    if not dataset:
        raise Exception("No active dataset found for this session.")
    return dataset

def get_embedding_model_for_dataset(dataset):
    set_api_keys_from_settings()
    return get_embedding_model(dataset.embedding_model_type, dataset.embedding_model_name)

def get_ranker(dataset, embedding_model, embedding_weighting_enable, embedding_weights, search_dimensions=None):
    if embedding_weighting_enable:
        # DecomposedRanker expects weights as a dict
        return DecomposedRanker(dataset, embedding_model, weights=embedding_weights, search_dimensions=search_dimensions)
    else:
        return SimpleRanker(dataset, embedding_model)

def get_llm_reranker(llm_type, mode, weights, batch_size=5):
    set_api_keys_from_settings()
    llm = LLM()
    if llm_type == "image":
        return LLMRerankerImage(llm, mode=mode, weights=weights, batch_size=batch_size)
    else:
        return LLMRerankerText(llm, mode=mode, weights=weights, batch_size=batch_size)

def format_results(results, embedding_results=None):
    # embedding_results: original embedding-based RankedUserInterfaceDocument list (optional)
    embedding_scores_map = {}
    if embedding_results is not None:
        for r in embedding_results:
            if hasattr(r.doc, 'id') and r.doc.id is not None:
                embedding_scores_map[r.doc.id] = r.dimension_scores if hasattr(r, 'dimension_scores') else {}
    formatted = []
    for result in results:
        doc = result.doc
        # LLM breakdown (from reranking)
        llm_breakdown = None
        if embedding_results is not None and hasattr(result, 'dimension_scores') and result.dimension_scores:
            llm_breakdown = {k: max(round(v * 100, 2), 0) for k, v in result.dimension_scores.items()}
        # Embedding breakdown (from embedding retrieval)
        embedding_breakdown = None
        if embedding_scores_map and hasattr(doc, 'id') and doc.id in embedding_scores_map:
            embedding_breakdown = {k: max(round(v * 100, 2), 0) for k, v in embedding_scores_map[doc.id].items()}
        # If only embedding is used, set scoring_breakdown to a dict and llm_breakdown to None
        if embedding_results is None:
            if hasattr(result, 'dimension_scores') and result.dimension_scores:
                embedding_breakdown = {k: max(round(v * 100, 2), 0) for k, v in result.dimension_scores.items()}
            else:
                embedding_breakdown = {}
            llm_breakdown = None
        formatted.append({
            'name': getattr(doc, 'text', getattr(doc, 'filepath', 'Result')),
            'score': "{:.2f}%".format(result.score * 100),
            'rank': result.rank,
            'source': result.source,
            'image': doc.filepath,
            'annotation': doc.annotation,
            'scoring_breakdown': embedding_breakdown,
            'llm_breakdown': llm_breakdown,
        })
    print(f"Formatted results: {formatted}")
    return formatted

def parse_weights(data, prefix, request=None):
    # Helper to parse weights from request data
    dataset_id = request.session.get('active_dataset_id') if request else None
    if dataset_id:
        try:
            django_dataset = DjangoDataset.objects.get(id=dataset_id)  # type: ignore[attr-defined]
            dims = DjangoSearchDimension.objects.filter(dataset=django_dataset)  # type: ignore[attr-defined]
            keys = [d.name for d in dims if d.query_decomposition]
        except DjangoDataset.DoesNotExist:  # type: ignore[attr-defined]
            keys = [d.name for d in DEFAULT_SEARCH_DIMENSIONS if d.query_decomposition]
    else:
        keys = [d.name for d in DEFAULT_SEARCH_DIMENSIONS if d.query_decomposition]
    weights = {}
    print("parsing weights")
    print(f"Keys: {keys}")
    print(f"Data: {data}")
    print(f"Prefix: {prefix}")
    for k in keys:
        val = data.get(f'{prefix}_weight_{k}')
        if val is not None:
            try:
                weights[k if k != 'uicomponents' else 'gui_components'] = float(val)
            except Exception:
                pass
    return weights

@csrf_exempt
def search_api(request):
    print(f"Search API called")
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=405)
    try:
        data = json.loads(request.body)
    except Exception as e:
        print(f"Error parsing request body: {e}")
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    # Read and parse parameters
    query = data.get('query', '')
    embedding_topk = int(data.get('embedding_topk', 100))
    embedding_threshold = float(data.get('embedding_threshold', 0.0))
    embedding_weighting_enable = str(data.get('embedding_weighting_enable', 'false')).lower() in ['true', '1', 'yes', 'on']
    embedding_weights = parse_weights(data, 'embedding', request) if embedding_weighting_enable else None
    llm_enable = str(data.get('llm_enable', 'false')).lower() in ['true', '1', 'yes', 'on']
    llm_topk = int(data.get('llm_topk', embedding_topk))
    llm_threshold = float(data.get('llm_threshold', embedding_threshold))
    llm_model = data.get('llm_model', 'gpt-4.1')
    llm_type = data.get('llm_type', 'text')
    llm_temp = float(data.get('llm_temp', 0.5))
    llm_weighting_enable = str(data.get('llm_weighting_enable', 'false')).lower() in ['true', '1', 'yes', 'on']
    llm_weights = parse_weights(data, 'llm', request) if llm_weighting_enable else None

    # 1. Load dataset and embedding model
    try:
        dataset = get_active_framework_dataset(request)
    except Exception as e:
        return JsonResponse({'error': "No dataset selected. Please select a dataset first."}, status=400)
    embedding_model = get_embedding_model_for_dataset(dataset)
    print(f"Dataset: {dataset}")
    print(f"Embedding model: {embedding_model}")

    # Fetch and update search dimensions from DB
    dataset_id = request.session.get('active_dataset_id')
    if dataset_id:
        try:
            django_dataset = DjangoDataset.objects.get(id=dataset_id)  # type: ignore[attr-defined]
            dims = DjangoSearchDimension.objects.filter(dataset=django_dataset)  # type: ignore[attr-defined]
            search_dimensions = [searchdimension_from_model(d) for d in dims if d.query_decomposition]
        except DjangoDataset.DoesNotExist:  # type: ignore[attr-defined]
            search_dimensions = [d for d in DEFAULT_SEARCH_DIMENSIONS if d.query_decomposition]
    else:
        search_dimensions = [d for d in DEFAULT_SEARCH_DIMENSIONS if d.query_decomposition]
    # Update weights from user settings if provided
    if embedding_weighting_enable and embedding_weights:
        for dim in search_dimensions:
            if dim.name in embedding_weights:
                dim.weight = embedding_weights[dim.name]

    print("Embedding-retrieval")
    print(f"Embedding-retrieval params: {embedding_weighting_enable}, {embedding_weights}")
    # 2. Embedding-retrieval
    ranker = get_ranker(dataset, embedding_model, embedding_weighting_enable, embedding_weights, search_dimensions=search_dimensions)
    print(f"Ranker: {ranker}")
    ranked_results = ranker.rank(query, conf_threshold=embedding_threshold, top_k=embedding_topk)
    print(f"Ranked results: {ranked_results}")
    # Attach image URLs and annotations for frontend
    for result in ranked_results:
        doc = result.doc
        try:
            image_entry = ImageEntry.objects.get(id=doc.id)
            image_url = image_entry.file.url if image_entry.file else ''
            annotation = json.loads(image_entry.annotation)
            doc.annotation = annotation
            doc.filepath = image_url
            try:
                doc.text = doc.filepath.split('/')[-1].split('.')[0]
            except Exception:
                doc.text = f"GUI {doc.id}"
        except ImageEntry.DoesNotExist:
            doc.filepath = ''
            doc.annotation = None
    print("Embedding-retrieval done")
    print(f"Ranked results: {ranked_results}")
    # 3. LLM reranking (optional)
    if not llm_enable:
        return JsonResponse({'results': format_results(ranked_results)})

    # Prepare reranker mode and weights
    reranker_mode = LLMRerankerText.MODE_DECOMPOSED if llm_weighting_enable else LLMRerankerText.MODE_SINGLE_SCORE
    reranker_weights = llm_weights if llm_weighting_enable else None
    batch_size = BATCH_SIZE_CELERY_SEARCH
    print("LLM reranking")
    print(f"LLM reranking params: {llm_weighting_enable}, {llm_weights}")
    # Collect all LLM reranker parameters into a dict
    print(f"LLM model: {llm_model}")
    llm_reranker_params = {
        'llm_model': llm_model,
        'llm_type': llm_type,
        'llm_temp': llm_temp,
        'llm_weighting_enable': llm_weighting_enable,
        'llm_weights': llm_weights,
        'reranker_mode': reranker_mode,
        'reranker_weights': reranker_weights,
        'llm_topk': llm_topk,
        'llm_threshold': llm_threshold,
        'dataset_id': dataset_id,
    }
    llm_reranker_params_json = json.dumps(llm_reranker_params)
    print(f"LLM reranking params: {llm_reranker_params_json}")
    # Celery batching for reranking
    ranked_results_dicts = [r.to_dict() for r in ranked_results]
    batches = split_into_batches(ranked_results_dicts, batch_size)
    print(f"Batch-length: {len(batches)}")
    # Each batch needs to be called with (batch, query, llm_reranker_params_json)
    job = group(rerank_gui_batch.s(batch, query, llm_reranker_params_json) for batch in batches)()
    results_batches = job.get(timeout=CELERY_SEARCH_TIMEOUT)
    reranked_dicts = merge_reranked_results(results_batches)
    # Convert back to RankedUserInterfaceDocument objects
    reranked_results = [RankedUserInterfaceDocument.from_dict(d) for d in reranked_dicts]
    # Attach image URLs and annotations again (in case ranks changed)
    for result in reranked_results:
        doc = result.doc
        try:
            image_entry = ImageEntry.objects.get(id=doc.id)
            image_url = image_entry.file.url if image_entry.file else ''
            annotation = json.loads(image_entry.annotation)
            doc.annotation = annotation
            doc.filepath = image_url
            try:
                doc.text = doc.filepath.split('/')[-1].split('.')[0]
            except Exception:
                doc.text = f"GUI {doc.id}"
        except ImageEntry.DoesNotExist:
            doc.filepath = ''
            doc.annotation = None
    print("LLM reranking done")
    print(f"Reranked results: {reranked_results}")
    return JsonResponse({'results': format_results(reranked_results, embedding_results=ranked_results)})


def split_into_batches(lst: List[dict], batch_size: int) -> List[List[dict]]:
    """Split a list of dicts into batches of batch_size."""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

def merge_reranked_results(results_batches: List[List[dict]]) -> List[dict]:
    all_results = [item for batch in results_batches for item in batch]
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    for idx, item in enumerate(all_results, 1):
        item['rank'] = idx
    return all_results