from django.shortcuts import render
from gui_rerank.llm.llm import LLM
from .models import Dataset
from django.shortcuts import redirect
from django.urls import reverse
from django import forms
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import get_object_or_404
from .tasks import build_dataset
import time
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import json
from django.core.cache import cache
from .adapters import django_dataset_to_framework_dataset
from gui_rerank.llm.config import AVAILABLE_MODELS
from gui_rerank.embeddings.config import AVAILABLE_EMBEDDING_MODELS
from gui_rerank.annotator.config import DEFAULT_SEARCH_DIMENSIONS
from gui_rerank.models.search_dimension import SearchDimension as SearchDimensionDC
from gui_rerank.annotator.config import create_general_description_dimension
from settings.models import set_api_keys_from_settings
__all__ = ('mock_long_task',)


class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'zip_file']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control form-control-lg rounded-3 w-100'}),
            'zip_file': forms.ClearableFileInput(attrs={'class': 'form-control', 'accept': '.zip'}),
        }


def dataset_list(request):
    datasets = Dataset.objects.all().order_by('-created_at')  # type: ignore[attr-defined]
    return render(request, 'datasets/dataset_list.html', {'datasets': datasets})


def add_dataset(request):
    set_api_keys_from_settings()
    print("add_dataset")
    # LLM and embedding model options
    llm_options = AVAILABLE_MODELS
    embedding_options = AVAILABLE_EMBEDDING_MODELS
    batch_size_options = [1, 3, 5, 8, 10, 15]
    default_llm = LLM.MODEL_GPT_4_1
    default_embedding_type, default_embedding_name, _ = AVAILABLE_EMBEDDING_MODELS[0]
    # Prepare default search dimensions for frontend
    default_search_dimensions = [
        {"name": d.name, "annotation_description": d.annotation_description}
        for d in DEFAULT_SEARCH_DIMENSIONS if d.name != "general_description"
    ]
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        # Read extra fields from POST
        llm_model = request.POST.get('llm_model', default_llm)
        embedding_model_type = request.POST.get('embedding_model_type', default_embedding_type)
        embedding_model_name = request.POST.get('embedding_model_name', default_embedding_name)
        batch_size = int(request.POST.get('batch_size', 5))
        search_dimensions_json = request.POST.get('search_dimensions_json', '[]')
        try:
            user_dims = json.loads(search_dimensions_json)
        except Exception:
            user_dims = []
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.annotation_model_name = llm_model
            dataset.embedding_model_type = embedding_model_type
            dataset.embedding_model_name = embedding_model_name
            dataset.batch_size = batch_size
            dataset.state = 'pending'
            dataset.save()
            # Save search dimensions
            from .models import SearchDimension as DjangoSearchDimension
            # Create dataclass objects for user dims
            dim_objs = [
                SearchDimensionDC(
                    name=d['name'],
                    annotation_description=d['annotation_description']
                ) for d in user_dims if d.get('name') and d.get('annotation_description')
            ]
            # Create general_description dimension
            if dim_objs:
                general_dim = create_general_description_dimension(dim_objs)
                dim_objs.append(general_dim)
            # Save all to DB
            for dim in dim_objs:
                DjangoSearchDimension.objects.create(
                    dataset=dataset,
                    name=dim.name,
                    annotation_description=dim.annotation_description,
                    type=dim.type,
                    query_decomposition=dim.query_decomposition,
                    negation=dim.negation,
                    weight=dim.weight,
                    pos_weight=dim.pos_weight,
                    neg_weight=dim.neg_weight
                )
            print(f"Dataset saved: {dataset.id}")
            # Pass all user choices to build_dataset
            result = build_dataset.delay(dataset.id, llm_model, embedding_model_type, embedding_model_name, batch_size)
            print(f"Result: {result}")
            print(f"result.status: {result.status}")
            return redirect(reverse('dataset_list'))
    else:
        form = DatasetForm()
    return render(request, 'datasets/add_dataset.html', {
        'form': form,
        'llm_options': llm_options,
        'embedding_options': embedding_options,
        'batch_size_options': batch_size_options,
        'default_llm': default_llm,
        'default_embedding_type': default_embedding_type,
        'default_embedding_name': default_embedding_name,
        'default_search_dimensions': json.dumps(default_search_dimensions),
    })


def search_view(request, dataset_id=None):
    # Retrieve the cached framework dataset for this user/session
    cache_key = f"active_framework_dataset_{request.session.session_key}"
    framework_dataset = cache.get(cache_key)
    # Use framework_dataset for ranking/searching
    # ... rest of your search logic ...
    return HttpResponse(f"Search view for dataset {dataset_id} (to be implemented)")

def edit_dataset(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    class EditDatasetForm(forms.ModelForm):
        class Meta:
            model = Dataset
            fields = ['name']  # Only allow editing metadata fields
            widgets = {
                'name': forms.TextInput(attrs={'class': 'form-control form-control-lg rounded-3'}),
            }
    # Efficiently fetch up to 10 example images ordered by id
    example_images = dataset.images.order_by('id')[:10]
    examples = []
    for img in example_images:
        try:
            import json
            annotation = json.loads(img.annotation) if img.annotation else {}
        except Exception:
            annotation = img.annotation  # fallback to raw string
        examples.append({
            'url': img.file.url,
            'name': img.name,
            'annotation': annotation,
        })
    # Group examples into pairs for the carousel
    def grouper(lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]
    example_pairs = grouper(examples, 2)
    if request.method == 'POST':
        form = EditDatasetForm(request.POST, instance=dataset)
        if form.is_valid():
            form.save()
            return redirect('dataset_list')
    else:
        form = EditDatasetForm(instance=dataset)
    return render(request, 'datasets/edit_dataset.html', {'form': form, 'dataset': dataset, 'example_pairs': example_pairs})

def dataset_list_api(request):
    datasets = Dataset.objects.all().order_by('-created_at')
    data = []
    for d in datasets:
        images = d.images.order_by('id').only('file')[:6]
        example_images = [img.file.url for img in images]
        num_images = d.images.count()
        # Fetch tags from SearchDimension for this dataset
        tags = list(d.search_dimensions.filter(query_decomposition=True).values_list('name', flat=True))
        data.append({
            'id': d.id,
            'name': d.name,
            'created_at': d.created_at.strftime('%Y-%m-%d %H:%M'),
            'zip_file': d.zip_file.url if d.zip_file else None,
            'state': d.state,
            'example_images': example_images,
            'num_images': num_images,
            'tags': tags,  # Now per-dataset tags from DB
            'error_message': d.error_message,
        })
    return JsonResponse({'datasets': data})

@csrf_exempt  # For demo; use CSRF protection in production
def set_active_dataset(request):
    if request.method == 'POST':
        dataset_id = request.POST.get('dataset_id')
        if dataset_id:
            request.session['active_dataset_id'] = dataset_id
            # Load and cache the framework dataset for this user/session
            from .models import Dataset as DjangoDataset
            django_dataset = DjangoDataset.objects.get(id=dataset_id)
            framework_dataset = django_dataset_to_framework_dataset(django_dataset)
            # Cache per user/session (can use user id or session key)
            cache_key = f"active_framework_dataset_{request.session.session_key}"
            cache.set(cache_key, framework_dataset, timeout=60*60)  # 1 hour
            return redirect('/search')
    return redirect('/')

@csrf_exempt
def delete_dataset(request, dataset_id):
    if request.method == 'POST':
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            dataset.delete()
            return HttpResponse(status=204)
        except Dataset.DoesNotExist:
            return HttpResponseNotFound('Dataset not found')
    return HttpResponse(status=405)
