import os
from typing import List
from gui_rerank.models.screen_image import ScreenImage

class DataLoader:

    @staticmethod
    def load_screen_images(path: str) -> List[ScreenImage]:
        images = []
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_id = os.path.splitext(fname)[0]
                images.append(ScreenImage(id=img_id, filelocation=fpath))
        return images


if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/examples/small'))
    images = DataLoader.load_screen_images(base)
    for img in images:
        print(img) 