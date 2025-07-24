import os
from gui_rerank.models.screen_image import ScreenImage
from gui_rerank.ranking.dataset_builder import DatasetBuilder
from gui_rerank.models.dataset import Dataset

class MockAnnotator:
    def annotate_batch(self, paths):
        # Fail on a specific file
        return [{} if "fail" not in p else (_ for _ in ()).throw(Exception("fail")) for p in paths]

class MockEmbedder:
    def __init__(self):
        self.model_type = "mock"
        self.model_name = "mock"

    def embed_documents(self, texts):
        # Fail on a specific text
        if any("fail" in t for t in texts):
            raise Exception("fail")
        return [[0.0] * 10 for _ in texts]

images = [
    ScreenImage(id="1", filelocation="good1.jpg"),
    ScreenImage(id="2", filelocation="fail.jpg"),
    ScreenImage(id="3", filelocation="good2.jpg"),
]
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/datasets/'))
checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/checkpoints/'))
builder = DatasetBuilder(MockAnnotator(), MockEmbedder())
dataset = builder.create(images, name="test", dataset_path=dataset_path, checkpoint_path=checkpoint_path, batch_size=1)
dataset.save()
print(dataset)
assert "fail.jpg" in dataset.failed_files