from django.core.management.base import BaseCommand
from gui_rerank.embeddings.lang_chain_embeddings import LangChainEmbeddings
from gui_rerank.llm.llm import LLM
from gui_rerank.models.dataset import Dataset as FrameworkDataset
from datasets.models import Dataset as DjangoDataset, ImageEntry
import os
import json
import shutil
from pathlib import Path
from django.conf import settings
from gui_rerank.annotator.config import DEFAULT_SEARCH_DIMENSIONS
from datasets.models import SearchDimension as DjangoSearchDimension

class Command(BaseCommand):
    help = "Import a merged RICO dataset into the Django database and create ImageEntry objects. Provide a dataset folder with --folder and a dataset name with --name."

    def add_arguments(self, parser):
        parser.add_argument('--folder', type=str, required=True, help='Folder name of the dataset to import (e.g., rico_filtered_merged)')
        parser.add_argument('--name', type=str, required=True, help='Name for the imported dataset in the database')

    def handle(self, *args, **options):
        # 1. Load the framework dataset
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../gui_rerank/resources/datasets/'))
        dataset_name = options['folder']
        framework_dataset = FrameworkDataset.load(dataset_dir, name=dataset_name)
        print(framework_dataset)
        # 2. Create Django Dataset object
        django_dataset = DjangoDataset.objects.create(
            name=options['name'],
            annotation_model_name=LLM.MODEL_GPT_4_1,
            embedding_model_name=LangChainEmbeddings.OPENAI_TEXT_EMBEDDING_LARGE_3,
            embedding_model_type=LangChainEmbeddings.MODEL_TYPE,
            state="success",
        )
        django_dataset.save()
        # 2.1. Create SearchDimension objects for this dataset using DEFAULT_SEARCH_DIMENSIONS
        for dim in DEFAULT_SEARCH_DIMENSIONS:
            DjangoSearchDimension.objects.create(
                dataset=django_dataset,
                name=dim.name,
                annotation_description=dim.annotation_description,
                type=dim.type,
                query_decomposition=dim.query_decomposition,
                negation=dim.negation,
                weight=dim.weight,
                pos_weight=dim.pos_weight,
                neg_weight=dim.neg_weight
            )
        # 3. Create ImageEntry objects and use ImageEntry.id as index
        new_annotations = {}
        new_index_mapping = {}
        new_inverse_index_mapping = {}
        rico_to_imageentry = {}
        for rico_id, annotation in framework_dataset.annotations.items():
            
            # Create ImageEntry (customize fields as needed)
            image_entry = ImageEntry.objects.create(
                dataset=django_dataset,
                name=rico_id+".jpg",
                file=f"dataset_images/{django_dataset.id}/{rico_id}.jpg",
                annotation=json.dumps(annotation),
            )
            rico_to_imageentry[rico_id] = image_entry.id
            new_annotations[image_entry.id] = annotation
            new_inverse_index_mapping[image_entry.id] = framework_dataset.inverse_index_mapping[rico_id]
        for key, value in framework_dataset.index_mapping.items():
            new_index_mapping[key] = rico_to_imageentry[value]

        # Save new files or update DB as needed
        # Example: save new annotation/index mapping as JSON files
        output_dir = os.path.join(dataset_dir, dataset_name)
        with open(os.path.join(output_dir, 'annotations_db.json'), 'w', encoding='utf-8') as f:
            json.dump(new_annotations, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, 'index_mapping_db.json'), 'w', encoding='utf-8') as f:
            json.dump(new_index_mapping, f, indent=2, ensure_ascii=False)

        # 5. Save all dataset files to /media/dataset_dataset/<dataset_id>/ using the Dataset.save method
        dataset_media_dir_base = Path(settings.MEDIA_ROOT) / 'dataset_dataset'
        dataset_media_dir = dataset_media_dir_base / str(django_dataset.id)
        dataset_media_dir.mkdir(parents=True, exist_ok=True)
        # Create a new framework dataset with the new annotations and index mapping
        framework_dataset.annotations = new_annotations
        framework_dataset.index_mapping = new_index_mapping
        framework_dataset.inverse_index_mapping = new_inverse_index_mapping
        framework_dataset.name = str(django_dataset.id)
        framework_dataset.save(base_path=str(dataset_media_dir_base))

        # 6. Copy each included RICO image to /media/dataset_images/<dataset_id>/
        source_image_dir = Path(r"E:/ZBook Juli 2025/Data/workspace_python/GUI2R/webapp/gui2rapp/staticfiles/resources/combined")
        images_media_dir = Path(settings.MEDIA_ROOT) / 'dataset_images' / str(django_dataset.id)
        images_media_dir.mkdir(parents=True, exist_ok=True)
        num_copied = 0
        for rico_id, imageentry_id in rico_to_imageentry.items():
            src_img = source_image_dir / f"{rico_id}.jpg"
            dst_img = images_media_dir / f"{rico_id}.jpg"
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                num_copied += 1
            else:
                self.stdout.write(self.style.WARNING(f"Source image not found: {src_img}"))

        self.stdout.write(self.style.SUCCESS(f"Copied {num_copied} images to {images_media_dir}"))
        self.stdout.write(self.style.SUCCESS(f"Dataset files saved to {dataset_media_dir}"))

        self.stdout.write(self.style.SUCCESS(f"Imported dataset '{dataset_name}' and created {len(rico_to_imageentry)} ImageEntries.")) 