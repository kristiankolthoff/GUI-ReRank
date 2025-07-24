import os
import pytest
from unittest.mock import patch
from gui_rerank.annotator.llm_annotator import LLMAnnotator
from gui_rerank.llm.llm import LLM

@pytest.fixture
def example_image_path():
    # Use one of the example images
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../resources/examples/small/30889.jpg'))

@pytest.fixture
def example_image_paths():
    # Use three example images
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '../resources/examples/small'))
    return [
        os.path.join(base, '30889.jpg'),
        os.path.join(base, '27400.jpg'),
        os.path.join(base, '25203.jpg'),
    ]

def test_single_annotation(example_image_path):
    fixed_json = '{"domain": "test", "functionality": "test", "design": "test", "low_level_features": ["a"], "gui_components": ["b"], "text_elements": ["c"]}'
    with patch.object(LLM, 'invoke', return_value=fixed_json):
        annotator = LLMAnnotator(llm=LLM())
        result = annotator.annotate(example_image_path)
        assert isinstance(result, dict)
        assert 'domain' in result
        assert 'functionality' in result
        assert 'design' in result
        assert 'low_level_features' in result
        assert 'gui_components' in result
        assert 'text_elements' in result

def test_batch_annotation(example_image_paths):
    # Simulate a batch response with ids '1', '2', '3'
    batch_json = '{"1": {"domain": "test", "functionality": "test", "design": "test", "low_level_features": ["a"], "gui_components": ["b"], "text_elements": ["c"]}, "2": {"domain": "test", "functionality": "test", "design": "test", "low_level_features": ["a"], "gui_components": ["b"], "text_elements": ["c"]}, "3": {"domain": "test", "functionality": "test", "design": "test", "low_level_features": ["a"], "gui_components": ["b"], "text_elements": ["c"]}}'
    with patch.object(LLM, 'invoke', return_value=batch_json):
        annotator = LLMAnnotator(llm=LLM(), batch_size=2)
        results = annotator.annotate_batch(example_image_paths)
        assert isinstance(results, list)
        assert len(results) == len(example_image_paths)
        for result in results:
            assert isinstance(result, dict)
            assert 'domain' in result
            assert 'functionality' in result
            assert 'design' in result
            assert 'low_level_features' in result
            assert 'gui_components' in result
            assert 'text_elements' in result 