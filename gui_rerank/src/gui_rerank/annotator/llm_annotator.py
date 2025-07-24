import base64
from typing import Any, Dict, List, Optional, Union
import os

from gui_rerank.annotator.annotator import Annotator
from gui_rerank.llm.llm import LLM

import json

from django.conf import settings
from gui_rerank.annotator.config import DEFAULT_SEARCH_DIMENSIONS
from gui_rerank.models.search_dimension import SearchDimension

class LLMAnnotator(Annotator):

    PLACEHOLDER_ANNOTATION_ELEMENT = "{placeholder_annotation_elements}"
    
    def _replace_annotation_elements_placeholder(self, prompt: str, search_dimensions: List[SearchDimension]) -> str:
        # Build the annotation_elements dict dynamically from search_dimensions
        annotation_elements = {d.name: d.annotation_description for d in search_dimensions}
        return prompt.replace(LLMAnnotator.PLACEHOLDER_ANNOTATION_ELEMENT, json.dumps(annotation_elements))

    def __init__(self, llm: LLM, prompt_path: Optional[str] = None, batch_prompt_path: Optional[str] = None, batch_size: Optional[int] = 5, search_dimensions: Optional[List[SearchDimension]] = None):
        self.llm = llm
        self.batch_size = int(batch_size) if batch_size is not None else 5
        self.search_dimensions = search_dimensions if search_dimensions is not None else DEFAULT_SEARCH_DIMENSIONS
        if prompt_path is None:
            prompt_path = os.path.join(os.path.dirname(__file__), '../../../resources/prompts/annotation_prompt.txt')
        if batch_prompt_path is None:
            batch_prompt_path = os.path.join(os.path.dirname(__file__), '../../../resources/prompts/annotation_batch_prompt.txt')
        with open(os.path.abspath(prompt_path), 'r', encoding='utf-8') as f:
            prompt_content = f.read()
            self.system_prompt = self._replace_annotation_elements_placeholder(prompt_content, self.search_dimensions)
            #print(self.system_prompt)
        with open(os.path.abspath(batch_prompt_path), 'r', encoding='utf-8') as f:
            batch_prompt_content = f.read()
            self.batch_prompt = self._replace_annotation_elements_placeholder(batch_prompt_content, self.search_dimensions)
            #print(self.batch_prompt)

    @staticmethod
    def encode_image(image_path: str) -> str:
        #print(f"Encoding image: {image_path}")
        image_path = image_path #os.path.join(settings.MEDIA_ROOT, image_path)
        #print(f"Encoded image path: {image_path}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_single_image_message(self, base64_img: str) -> Union[str, list]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this screenshot as instructed."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                ],
            },
        ]

    def annotate(self, image_path: str) -> Dict:
        base64_img = self.encode_image(image_path)
        messages = self._build_single_image_message(base64_img)
        response, _ = self.llm.invoke(messages)
        try:
            if isinstance(response, dict):
                return response
            response = response.strip().replace('```json', '').replace('```', '')
            import json
            return json.loads(response)
        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM response as JSON: {response}") from e

    def _build_batch_image_message(self, base64_imgs: List[str]) -> list:
        content = []
        for idx, base64_img in enumerate(base64_imgs, 1):
            content.append({"type": "text", "text": f"Image {idx}: Describe this screenshot as instructed. Use id '{idx}' in your output."})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
        return [
            {"role": "system", "content": self.batch_prompt},
            {"role": "user", "content": content}
        ]

    def annotate_batch(self, image_paths: List[str]) -> List[Dict]:
        results = []
        total = len(image_paths)
        for batch_start in range(0, total, self.batch_size):
            batch_paths = image_paths[batch_start:batch_start + self.batch_size]
            base64_imgs = [self.encode_image(path) for path in batch_paths]
            messages = self._build_batch_image_message(base64_imgs)
            response, _ = self.llm.invoke(messages)
            print(f"response: {response}")
            try:
                if isinstance(response, dict):
                    result_dict = response
                else:
                    response = response.strip().replace('```json', '').replace('```', '')
                    result_dict = json.loads(response)
                # The ids are 1-based indices as strings
                id_list = [str(idx) for idx in range(1, len(batch_paths) + 1)]
                #print("result_dict")
                #print(result_dict)
                results.extend([result_dict.get(img_id, {}) for img_id in id_list])
            except Exception as e:
                print(f"from failure method annotate batch")
                raise RuntimeError(f"Failed to parse LLM batch response as JSON: {response}") from e
        return results


if __name__ == "__main__":
    import os
    from gui_rerank.llm.llm import LLM

    # Example image paths
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../resources/examples/small'))
    image_paths = [
        os.path.join(base, '30889.jpg'),
        os.path.join(base, '27400.jpg'),
        os.path.join(base, '25203.jpg'),
    ]

    llm = LLM()
    annotator = LLMAnnotator(llm=llm, batch_size=3)

    print("Single annotation result:")
    single_result = annotator.annotate(image_paths[1])
    print(single_result)

    # print("\nBatch annotation result:")
    # batch_result = annotator.annotate_batch(image_paths)
    # print(batch_result)

