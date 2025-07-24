import numpy as np
import os
import pickle
import shutil
from typing import List, Callable, Dict, Any, Optional, Tuple, Union
from gui_rerank.models.screen_image import ScreenImage
from gui_rerank.annotator.llm_annotator import LLMAnnotator
from gui_rerank.models.dataset import Dataset
from gui_rerank.embeddings.embeddings import Embeddings
from gui_rerank.annotator.config import DEFAULT_SEARCH_DIMENSIONS
from gui_rerank.models.search_dimension import SearchDimension
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError


class DatasetBuilder:
    def __init__(self, annotator: LLMAnnotator, embedding_model: Embeddings) -> None:
        self.annotator = annotator
        self.embedding_model = embedding_model
        self.failed_files = []

    def _save_checkpoint(
        self,
        checkpoint_dir: str,
        batch_idx: int,
        all_annotations: Dict[str, Any],
        all_ids: List[str],
        embedding_batches: Dict[str, List[np.ndarray]],
        failed_files: list
    ) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save annotations and ids as pickle (always overwrite)
        with open(os.path.join(checkpoint_dir, "annotations.pkl"), "wb") as f:
            pickle.dump(all_annotations, f)
        with open(os.path.join(checkpoint_dir, "ids.pkl"), "wb") as f:
            pickle.dump(all_ids, f)
        # Save each embedding batch as pickle (always overwrite)
        for key, batches in embedding_batches.items():
            with open(os.path.join(checkpoint_dir, f"{key}_batches.pkl"), "wb") as f:
                pickle.dump(batches, f)
        # Save failed files as pickle
        with open(os.path.join(checkpoint_dir, "failed_files.pkl"), "wb") as f:
            pickle.dump(failed_files, f)
        # Save a meta file for resume (always overwrite)
        with open(os.path.join(checkpoint_dir, "meta.pkl"), "wb") as f:
            pickle.dump({"last_batch": batch_idx}, f)

    def _load_latest_checkpoint(
        self,
        checkpoint_dir: str,
        search_dimensions: Optional[List[SearchDimension]] = None
    ) -> Optional[Tuple[int, Dict[str, Any], List[str], Dict[str, List[np.ndarray]], list]]:
        if search_dimensions is None:
            search_dimensions = DEFAULT_SEARCH_DIMENSIONS
        if not os.path.exists(checkpoint_dir):
            return None
        meta_path = os.path.join(checkpoint_dir, "meta.pkl")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        batch_idx = meta.get("last_batch", -1)
        if batch_idx < 0:
            return None
        # Load annotations and ids
        with open(os.path.join(checkpoint_dir, "annotations.pkl"), "rb") as f:
            all_annotations = pickle.load(f)
        with open(os.path.join(checkpoint_dir, "ids.pkl"), "rb") as f:
            all_ids = pickle.load(f)
        # Load embedding batches (use pickle)
        embedding_batches: Dict[str, List[np.ndarray]] = {d.name: [] for d in search_dimensions if d.type == 'embedding' and d.name is not None}
        for key in embedding_batches:
            pkl_path = os.path.join(checkpoint_dir, f"{key}_batches.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    embedding_batches[key] = pickle.load(f)
        # Load failed files if present
        failed_files_path = os.path.join(checkpoint_dir, "failed_files.pkl")
        if os.path.exists(failed_files_path):
            with open(failed_files_path, "rb") as f:
                failed_files = pickle.load(f)
        else:
            failed_files = []
        return batch_idx, all_annotations, all_ids, embedding_batches, failed_files

    def _cleanup_checkpoints(self, checkpoint_dir: str) -> None:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

    def create(
        self,
        screen_images: List[ScreenImage],
        name: str,
        dataset_path: str,
        checkpoint_path: str,
        progress_callback: Optional[Callable[[int, str, str], None]] = None,
        batch_size: int = 5,
        search_dimensions: Optional[List[SearchDimension]] = None
    ) -> Dataset:
        if search_dimensions is None:
            search_dimensions = DEFAULT_SEARCH_DIMENSIONS
        # Always use a subdirectory for checkpoints with the dataset name
        checkpoint_dir = os.path.join(checkpoint_path, name)
        # Try to resume from checkpoint
        resume = self._load_latest_checkpoint(checkpoint_dir, search_dimensions)
        if resume:
            last_batch, all_annotations, all_ids, embedding_batches, failed_files = resume
            self.failed_files = failed_files
            start_idx = (last_batch + 1) * batch_size
            print(f"Resuming from checkpoint batch {last_batch}, starting at image {start_idx}")
        else:
            all_annotations = {}
            all_ids = []
            embedding_batches: Dict[str, List[np.ndarray]] = {d.name: [] for d in search_dimensions if d.type == 'embedding' and d.name is not None}
            start_idx = 0
            self.failed_files = []
        num_images = len(screen_images)
        for batch_start in range(start_idx, num_images, batch_size):
            if progress_callback:
                progress = int(batch_start * 100 / num_images)
                progress_message = f"Processing Batch {int(batch_start/batch_size)} / {int(num_images/batch_size)}"
                progress_callback(progress, progress_message, name)
            batch_idx = batch_start // batch_size
            progress_message = f"Processing batch {batch_start} to {min(batch_start+batch_size, num_images)}"
            print(progress_message)
            batch = screen_images[batch_start:batch_start+batch_size]
            batch_paths = [img.filelocation for img in batch]
            batch_ids = [img.id for img in batch]
            try:
                batch_annotations = self.annotator.annotate_batch(batch_paths)
            except Exception as e:
                print(f"Annotation failed for batch {batch_paths}: {e}")
                message = str(e)
                if "API key not valid" in message or "API_KEY_INVALID" in message or "api_key" in message.lower():
                    raise ValueError(message) from e
                self.failed_files.extend(batch_paths)
                continue
            for img_id, annotation in zip(batch_ids, batch_annotations):
                all_annotations[img_id] = annotation
                all_ids.append(img_id)
            # Batch embedding for this batch
            for dim in search_dimensions:
                if dim.type == 'embedding' and dim.name is not None:
                    texts = [annotation.get(dim.name, "") for annotation in batch_annotations]
                    print(f"Batch embedding for {dim.name}: {texts}")
                    try:
                        emb = np.array(self.embedding_model.embed_documents(texts))
                        embedding_batches[dim.name].append(emb)
                    except Exception as e:
                        print(f"Embedding failed for batch {batch_paths} (key={dim.name}): {e}")
                        self.failed_files.extend(batch_paths)
                        continue
            # Save checkpoint after each batch
            self._save_checkpoint(checkpoint_dir, batch_idx, all_annotations, all_ids, embedding_batches, self.failed_files)
        # Stack all batches for each embedding dimension
        embeddings = {k: np.vstack(embedding_batches[k]) if embedding_batches[k] else np.zeros((0,)) for k in embedding_batches}
        # Build mappings
        index_mapping = {i: img_id for i, img_id in enumerate(all_ids)}
        inverse_index_mapping = {img_id: i for i, img_id in enumerate(all_ids)}
        # Create Dataset
        dataset = Dataset(
            name=name,
            path=dataset_path,
            index_mapping=index_mapping,
            inverse_index_mapping=inverse_index_mapping,
            annotations=all_annotations,
            embeddings=embeddings,
            embedding_model_type=self.embedding_model.model_type,
            embedding_model_name=self.embedding_model.model_name,
            failed_files=self.failed_files
        )
        # Clean up checkpoints
        self._cleanup_checkpoints(checkpoint_dir)
        return dataset


if __name__ == "__main__":
    import os
    from gui_rerank.loader.data_loader import DataLoader
    from gui_rerank.annotator.llm_annotator import LLMAnnotator
    from gui_rerank.llm.llm import LLM
    from gui_rerank.embeddings.lang_chain_embeddings import LangChainEmbeddings

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/examples/small'))
    images = DataLoader.load_screen_images(base)
    print(f"Loaded {len(images)} images.")

    # Custom search dimensions for demo/testing
    custom_search_dimensions = [
        SearchDimension(
            name="plattform",
            annotation_description="Name the plattform of the GUI screenshot.",
            type="embedding",
            query_decomposition=True,
            negation=True,
            weight=0.7,
            pos_weight=2.0,
            neg_weight=1.0
        ),
        SearchDimension(
            name="accessibility",
            annotation_description="Describe the accessibility aspects of the GUI screenshot.",
            type="embedding",
            query_decomposition=True,
            negation=False,
            weight=0.3,
            pos_weight=1.0,
            neg_weight=1.0
        )
    ]

    llm = LLM()
    annotator = LLMAnnotator(llm=llm, search_dimensions=custom_search_dimensions)
    embedding_model = LangChainEmbeddings(model_name=LangChainEmbeddings.OPENAI_TEXT_EMBEDDING_LARGE_3, dimensions=3072)
    builder = DatasetBuilder(annotator, embedding_model)
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/datasets/'))
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/checkpoints/'))
    print(f"Dataset path: {dataset_path}")
    print(f"Checkpoint path: {checkpoint_path}")
    dataset = builder.create(images, name="demo_2", dataset_path=dataset_path, 
                             checkpoint_path=checkpoint_path, search_dimensions=custom_search_dimensions, 
                             batch_size=3)
    print(dataset)
    dataset.save()
        


