import django
from gui_rerank.annotator.llm_annotator import LLMAnnotator
from gui_rerank.embeddings.lang_chain_embeddings import LangChainEmbeddings
from gui_rerank.dataset_builder.dataset_builder import DatasetBuilder

from .media_config import CHECKPOINTS_BASE_PATH, DATASETS_BASE_PATH
django.setup()

from celery import shared_task
import time
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import zipfile
import os
from django.conf import settings
from .models import Dataset, ImageEntry, SearchDimension as DjangoSearchDimension
from gui_rerank.llm.llm import LLM
from gui_rerank.query_decomposition.query_decomposition import QueryDecomposer
import json
from .adapters import imageentry_to_screenimage, searchdimension_from_model
from gui_rerank.embeddings.utils import get_embedding_model
from gui_rerank.annotator.config import SUPPORTED_IMAGE_FORMATS
from settings.models import set_api_keys_from_settings


@shared_task(bind=True)
def build_dataset(self, dataset_id: int, llm_model: str, embedding_model_type: str, embedding_model_name: str, batch_size: int = 5):
    set_api_keys_from_settings()
    try:
        print(f"Starting task for dataset {dataset_id}")
        print(f"LLM model: {llm_model}")
        # Fetch search dimensions for this dataset
        search_dim_models = DjangoSearchDimension.objects.filter(dataset_id=dataset_id)  # type: ignore[attr-defined]
        search_dimensions = [searchdimension_from_model(dim) for dim in search_dim_models]
        print(f"Search dimensions: {search_dimensions}")
        llm = LLM(model_name=llm_model)
        annotator = LLMAnnotator(llm=llm, search_dimensions=search_dimensions)
        print(f"Embedding model type: {embedding_model_type}")
        print(f"Embedding model name: {embedding_model_name}")
        embedding_model = get_embedding_model(model_type=embedding_model_type, model_name=embedding_model_name)
        print(f"Embedding model: {embedding_model}")
        builder = DatasetBuilder(annotator, embedding_model)
        #dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/datasets/'))
        dataset_path = os.path.join(settings.MEDIA_ROOT, DATASETS_BASE_PATH)
        #checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/checkpoints/'))
        checkpoint_path = os.path.join(settings.MEDIA_ROOT, CHECKPOINTS_BASE_PATH)
        channel_layer = get_channel_layer()
        group_name = f'dataset_progress_{dataset_id}'
        async_to_sync(channel_layer.group_send)(
            group_name,
            {
                'type': 'progress.update',
                'progress': 0,
                'message': 'Processing started'
            }
        ) 
        dataset = Dataset.objects.get(id=dataset_id)  # type: ignore[attr-defined]
        dataset.state = 'processing'
        dataset.save()
        zip_path = dataset.zip_file.path
        extract_dir = os.path.join(settings.MEDIA_ROOT, 'dataset_images', str(dataset_id))
        os.makedirs(extract_dir, exist_ok=True)
        image_entries = []
        screen_images = []
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist() if not f.endswith('/')]
            total = len(image_files)
            for i, file_name in enumerate(image_files, 1):
                # Only process supported image formats
                if not any(file_name.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS):
                    continue
                zip_ref.extract(file_name, extract_dir)
                # Save ImageEntry
                image_path = os.path.join('dataset_images', str(dataset_id), file_name)
                entry = ImageEntry.objects.create(
                    dataset=dataset,
                    name=os.path.basename(file_name),
                    file=image_path,
                    annotation=''  # Empty for now
                )  # type: ignore[attr-defined]
                image_entries.append(entry)
                screen_images.append(imageentry_to_screenimage(entry))
                # Progress update
                async_to_sync(channel_layer.group_send)(
                    group_name,
                    {
                        'type': 'progress.update',
                        'progress': int(i * 100 / total),
                        'message': f'Extracted {i}/{total} images...'
                    }
                )
        print(f"Image entries: {image_entries}")
        print(f"Screen images: {screen_images}")
        try:
            # When calling builder.create, pass search_dimensions
            dataset = builder.create(screen_images, name=str(dataset_id), dataset_path=dataset_path, 
                                     checkpoint_path=checkpoint_path, batch_size=batch_size, 
                                     search_dimensions=search_dimensions, progress_callback=progress_callback)
            print(dataset)
            dataset.save()
            # Save annotations to ImageEntry objects
            for img_id, annotation in dataset.annotations.items():
                try:
                    entry = ImageEntry.objects.get(id=img_id)
                    entry.annotation = json.dumps(annotation)
                    entry.save()
                except ImageEntry.DoesNotExist:
                    print(f"ImageEntry with id {img_id} does not exist.")
        except Exception as e:
            print(f"Exception during dataset build: {e}")
            dataset = Dataset.objects.get(id=dataset_id)
            dataset.state = 'failed'
            dataset.error_message = str(e)
            dataset.save()
            async_to_sync(channel_layer.group_send)(
                group_name,
                {
                    'type': 'progress.update',
                    'progress': 100,
                    'message': f'Failed: {str(e)}'
                }
            )
            return
        # Set is_processing to False and update base path when done
        db_dataset = Dataset.objects.get(id=dataset_id)
        db_dataset.state = 'success'
        db_dataset.save()
        async_to_sync(channel_layer.group_send)(
            group_name,
            {
                'type': 'progress.update',
                'progress': 100,
                'message': 'Processing finished'
            }
        )
    except Exception as e:
        print(f"Exception during dataset build (outer): {e}")
        dataset = Dataset.objects.get(id=dataset_id)
        dataset.state = 'failed'
        dataset.error_message = str(e)
        dataset.save()
        channel_layer = get_channel_layer()
        group_name = f'dataset_progress_{dataset_id}'
        async_to_sync(channel_layer.group_send)(
            group_name,
            {
                'type': 'progress.update',
                'progress': 100,
                'message': f'Failed: {str(e)}'
            }
        )
        # Do not re-raise

def progress_callback(progress: int, message: str, dataset_id: str):
    channel_layer = get_channel_layer()
    group_name = f'dataset_progress_{dataset_id}'
    async_to_sync(channel_layer.group_send)(
                group_name,
                {
                    'type': 'progress.update',
                    'progress': progress,
                    'message': message
                }
            )