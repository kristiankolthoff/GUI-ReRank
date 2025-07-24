from abc import ABC, abstractmethod
from typing import Dict, List

class Annotator(ABC):

    @abstractmethod
    def annotate(self, image_path: str) -> Dict:
        """
        Annotate a single image given its file path.
        Args:
            image_path (str): Path to the image file.
        Returns:
            Dict: Annotation result as a dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def annotate_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Annotate a batch of images given their file paths.
        Args:
            image_paths (List[str]): List of image file paths.
        Returns:
            List[Dict]: List of annotation results, one per image.
        """
        raise NotImplementedError
