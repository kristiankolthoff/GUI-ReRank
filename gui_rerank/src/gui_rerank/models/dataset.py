import numpy as np
import os
import json
from typing import Dict, Any, Optional, List

class Dataset:
    def __init__(self, name: str, path: str,
                 embeddings: Dict[str, np.ndarray],
                 index_mapping: Dict[int, str],
                 inverse_index_mapping: Dict[str, int],
                 annotations: Dict[str, Any],
                 embedding_model_type: str = "",
                 embedding_model_name: str = "",
                 failed_files: Optional[list] = None):
        self.name = name
        self.path = path
        self.embeddings = embeddings  # Dict[str, np.ndarray]
        self.index_mapping = index_mapping
        self.inverse_index_mapping = inverse_index_mapping
        self.annotations = annotations
        self.embedding_model_type = embedding_model_type
        self.embedding_model_name = embedding_model_name
        self.failed_files = failed_files or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'path': self.path,
            'index_mapping': self.index_mapping,
            'inverse_index_mapping': self.inverse_index_mapping,
            'annotations': self.annotations,
            'failed_files': self.failed_files,
            # Embeddings are not included in to_dict by default (can be added if needed)
        }

    def __repr__(self):
        emb_keys = list(self.embeddings.keys())
        emb_shapes = {k: v.shape for k, v in self.embeddings.items()}
        return (f"Dataset(name={self.name!r}, path={self.path!r}, "
                f"embeddings_keys={emb_keys}, embeddings_shapes={emb_shapes}, "
                f"index_mapping={len(self.index_mapping)}, inverse_index_mapping={len(self.inverse_index_mapping)}, "
                f"annotations={len(self.annotations)})")

    def save(self, base_path: Optional[str] = None) -> None:
        # Determine save directory
        save_dir = os.path.join(base_path or self.path, self.name)
        os.makedirs(save_dir, exist_ok=True)
        # Save embeddings
        for key, arr in self.embeddings.items():
            np.save(os.path.join(save_dir, f"{key}.npy"), arr)
        # Save annotations and mappings (pretty print)
        with open(os.path.join(save_dir, "annotations.json"), "w", encoding="utf-8") as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)
        with open(os.path.join(save_dir, "index_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(self.index_mapping, f, indent=2, ensure_ascii=False)
        with open(os.path.join(save_dir, "inverse_index_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(self.inverse_index_mapping, f, indent=2, ensure_ascii=False)
        # Save failed files
        with open(os.path.join(save_dir, "failed_files.json"), "w", encoding="utf-8") as f:
            json.dump(self.failed_files, f, indent=2, ensure_ascii=False)
        # Save meta
        meta = {
            "name": self.name,
            "path": self.path,
            "embedding_model_type": self.embedding_model_type,
            "embedding_model_name": self.embedding_model_name
        }
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, base_path: str, name: Optional[str] = None, embedding_model_type: Optional[str] = None,
             embedding_model_name: Optional[str] = None) -> "Dataset":
        print(f"Loading dataset from {base_path} with name {name}, embedding_model_type {embedding_model_type}, embedding_model_name {embedding_model_name}")
        # Determine load directory
        load_dir = os.path.join(base_path, name) if name else base_path
        # Load embeddings
        embeddings = {}
        for fname in os.listdir(load_dir):
            if fname.endswith(".npy"):
                key = fname[:-4]
                embeddings[key] = np.load(os.path.join(load_dir, fname))
        # Load annotations and mappings
        with open(os.path.join(load_dir, "annotations.json"), "r", encoding="utf-8") as f:
            annotations = json.load(f)
        with open(os.path.join(load_dir, "index_mapping.json"), "r", encoding="utf-8") as f:
            index_mapping = json.load(f)
            index_mapping = {int(k): v for k, v in index_mapping.items()}
        with open(os.path.join(load_dir, "inverse_index_mapping.json"), "r", encoding="utf-8") as f:
            inverse_index_mapping = json.load(f)
        # Load failed files
        failed_files_path = os.path.join(load_dir, "failed_files.json")
        if os.path.exists(failed_files_path):
            with open(failed_files_path, "r", encoding="utf-8") as f:
                failed_files = json.load(f)
        else:
            failed_files = []
        # Load meta
        with open(os.path.join(load_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(
            name=meta["name"],
            path=meta["path"],
            embeddings=embeddings,
            index_mapping=index_mapping,
            inverse_index_mapping=inverse_index_mapping,
            annotations=annotations,
            embedding_model_type=embedding_model_type or meta.get("embedding_model_type", ""),
            embedding_model_name=embedding_model_name or meta.get("embedding_model_name", ""),
            failed_files=failed_files
        )

    def merge_with(self, other: 'Dataset') -> 'Dataset':
        """
        Merge another Dataset into this one, updating embeddings, index mappings, and annotations.
        Returns a new merged Dataset.
        """
        import numpy as np
        # Merge embeddings (concatenate for each key)
        merged_embeddings = {}
        for key in set(self.embeddings.keys()).union(other.embeddings.keys()):
            arr1 = self.embeddings.get(key)
            arr2 = other.embeddings.get(key)
            if arr1 is not None and arr2 is not None:
                merged_embeddings[key] = np.concatenate([arr1, arr2], axis=0)
            elif arr1 is not None:
                merged_embeddings[key] = arr1.copy()
            elif arr2 is not None:
                merged_embeddings[key] = arr2.copy()

        # Merge index_mapping and inverse_index_mapping
        merged_index_mapping = {}
        merged_inverse_index_mapping = {}
        merged_annotations = {}
        # Start with self
        idx = 0
        for i in range(len(self.index_mapping)):
            img_id = self.index_mapping[i]
            merged_index_mapping[idx] = img_id
            merged_inverse_index_mapping[img_id] = idx
            if img_id in self.annotations:
                merged_annotations[img_id] = self.annotations[img_id]
            idx += 1
        # Then add other's, making sure to not duplicate ids
        for i in range(len(other.index_mapping)):
            img_id = other.index_mapping[i]
            if img_id not in merged_inverse_index_mapping:
                merged_index_mapping[idx] = img_id
                merged_inverse_index_mapping[img_id] = idx
                if img_id in other.annotations:
                    merged_annotations[img_id] = other.annotations[img_id]
                idx += 1
        # Merge meta info (take from self, but could be customized)
        merged_name = f"{self.name}_merged_{other.name}"
        merged_path = self.path
        merged_embedding_model_type = self.embedding_model_type
        merged_embedding_model_name = self.embedding_model_name
        return Dataset(
            name=merged_name,
            path=merged_path,
            embeddings=merged_embeddings,
            index_mapping=merged_index_mapping,
            inverse_index_mapping=merged_inverse_index_mapping,
            annotations=merged_annotations,
            embedding_model_type=merged_embedding_model_type,
            embedding_model_name=merged_embedding_model_name
        )

    @staticmethod
    def merge_many(datasets: list) -> 'Dataset':
        """
        Merge a list of Datasets into one, using merge_with in sequence.
        """
        if not datasets:
            raise ValueError("No datasets to merge.")
        merged = datasets[0]
        for ds in datasets[1:]:
            merged = merged.merge_with(ds)
        return merged
