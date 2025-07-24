import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from gui_rerank.llm.llm import LLM
from gui_rerank.annotator.config import DEFAULT_SEARCH_DIMENSIONS
from gui_rerank.models.search_dimension import SearchDimension

load_dotenv()

def _generate_expected_fields_and_default_response():
    expected_fields = []
    default_empty_response = {}
    for dim in DEFAULT_SEARCH_DIMENSIONS:
        if dim.query_decomposition:
            if dim.negation:
                expected_fields.append(dim.name)
                default_empty_response[dim.name] = {"pos": [], "neg": []}
            else:
                expected_fields.append(dim.name)
                default_empty_response[dim.name] = []
    return expected_fields, default_empty_response


@dataclass
class QueryDecomposerConfig:
    """Configuration for the QueryDecomposer class."""

    # LLM settings
    llm: Optional[LLM] = None
    
    # Prompt template path
    prompt_template_path: str = "resources/prompts/query_decomposition.txt"
    
    # Search dimensions
    search_dimensions: Optional[List[SearchDimension]] = field(default_factory=lambda: DEFAULT_SEARCH_DIMENSIONS.copy())
    
    # Output schema
    expected_fields: Optional[List[str]] = None
    # Error handling
    raise_on_json_error: bool = True
    default_empty_response: Optional[Dict[str, Any]] = None


class QueryDecomposer:
    """
    A class for decomposing natural language queries into structured filtering information
    for mobile UI screenshot search and ranking.
    """
    
    # Query placeholder constant in prompt template
    PLACEHOLDER_EXTRACTION_INFO = "{placeholder_extraction_info}"
    PLACEHOLDER_RELEVANT_INFO = "{placeholder_relevant_info}"
    QUERY_PLACEHOLDER = "{query_placeholder}"

    @staticmethod
    def _generate_expected_fields_and_default_response(search_dimensions: Optional[List[SearchDimension]] = None):
        if not search_dimensions:
            search_dimensions = DEFAULT_SEARCH_DIMENSIONS
        expected_fields = []
        default_empty_response = {}
        for dim in search_dimensions:
            if dim.query_decomposition:
                if dim.negation:
                    expected_fields.append(dim.name)
                    default_empty_response[dim.name] = {"pos": [], "neg": []}
                else:
                    expected_fields.append(dim.name)
                    default_empty_response[dim.name] = []
        return expected_fields, default_empty_response

    def __init__(self, config: Optional[QueryDecomposerConfig] = None, search_dimensions: Optional[List[SearchDimension]] = None):
        """
        Initialize the QueryDecomposer.
        Args:
            config: Configuration object. If None, uses default configuration.
            search_dimensions: Optional custom list of SearchDimension objects.
        """
        self.config = config or QueryDecomposerConfig()
        self.search_dimensions = search_dimensions or self.config.search_dimensions or DEFAULT_SEARCH_DIMENSIONS
        # Update config's search_dimensions for consistency
        self.config.search_dimensions = self.search_dimensions
        # Update expected_fields and default_empty_response based on search_dimensions
        expected_fields, default_empty_response = self._generate_expected_fields_and_default_response(self.search_dimensions)
        self.config.expected_fields = expected_fields
        self.config.default_empty_response = default_empty_response
        # Set up LLM
        if self.config.llm is None:
            self.config.llm = LLM(
                model_name=LLM.MODEL_GPT_4_1,
                temperature=0.05,
                max_tokens=5000
            )
        self.llm = self.config.llm
        # Load prompt template and replace extraction info placeholder
        prompt_template = self._load_prompt_template()
        extraction_info_json = self._extract_info()
        relevant_info = self._extract_relevant_info()
        self.prompt_template = prompt_template.replace(QueryDecomposer.PLACEHOLDER_EXTRACTION_INFO, extraction_info_json) \
                                              .replace(QueryDecomposer.PLACEHOLDER_RELEVANT_INFO, relevant_info)

    def _load_prompt_template(self) -> str:
        """
        Load the prompt template from file.
        
        Returns:
            The prompt template as a string
            
        Raises:
            FileNotFoundError: If the prompt template file doesn't exist
        """
        # Try to find the prompt template file
        template_paths = [
            Path(self.config.prompt_template_path),
            Path("gui_rerank") / self.config.prompt_template_path,
            Path(__file__).parent.parent / self.config.prompt_template_path,
            Path(__file__).parent.parent.parent / self.config.prompt_template_path,
            Path(__file__).parent.parent.parent.parent / self.config.prompt_template_path,
            Path(__file__).parent.parent.parent.parent.parent / self.config.prompt_template_path,
        ]
        
        for path in template_paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
        
        raise FileNotFoundError(f"Prompt template not found. Tried paths: {template_paths}")
    
    def decompose_query(self, query: str) -> Dict[str, List[str]]:
        """
        Decompose a natural language query into structured filtering information.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            # Replace placeholder with actual query in prompt template
            prompt = self.prompt_template.replace(QueryDecomposer.QUERY_PLACEHOLDER, query.strip())
            print("Prompt:")
            print(prompt)
            # Use the LLM to get response
            response, usage_metadata = self.llm.invoke(prompt)
            print("Response:")
            print(response)
            # Clean the response
            content = response.strip().replace("```json", "").replace("```", "")
            
            # Parse JSON response
            try:
                structured_response = json.loads(content)
            except json.JSONDecodeError as e:
                if self.config.raise_on_json_error:
                    raise Exception(f"Failed to parse JSON response: {content}") from e
                else:
                    print(f"Warning: Failed to parse JSON response: {content}")
                    if self.config.default_empty_response is not None:
                        return self.config.default_empty_response.copy()
                    else:
                        # fallback: generate from current search_dimensions
                        _, default_empty_response = self._generate_expected_fields_and_default_response(self.search_dimensions)
                        return default_empty_response.copy()
            
            # Validate and normalize response
            return self._normalize_response(structured_response)
            
        except Exception as e:
            if self.config.raise_on_json_error:
                raise Exception(f"Query decomposition failed: {str(e)}") from e
            else:
                print(f"Warning: Query decomposition failed: {str(e)}")
                if self.config.default_empty_response is not None:
                    return self.config.default_empty_response.copy()
                else:
                    _, default_empty_response = self._generate_expected_fields_and_default_response(self.search_dimensions)
                    return default_empty_response.copy()
    
    def _extract_info(self, search_dimensions: Optional[List[SearchDimension]] = None) -> str:
        if not search_dimensions:
            search_dimensions = self.search_dimensions
        results = {}
        for dim in search_dimensions:
            if dim.query_decomposition:
                if dim.negation:
                    results[dim.name] = {"pos": [], "neg": []}
                else:
                    results[dim.name] = []
        return json.dumps(results)

    def _extract_relevant_info(self, search_dimensions: Optional[List[SearchDimension]] = None) -> str:
        if not search_dimensions:
            search_dimensions = self.search_dimensions
        results = ", ".join([dim.name for dim in search_dimensions if dim.query_decomposition])
        return results

    def _normalize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize the response to ensure all expected fields are present and properly formatted.
        
        Args:
            response: Raw response from the API
            
        Returns:
            Normalized response with all expected fields
        """
        if self.config.default_empty_response is not None:
            normalized = self.config.default_empty_response.copy()
        else:
            _, default_empty_response = self._generate_expected_fields_and_default_response(self.search_dimensions)
            normalized = default_empty_response.copy()
        expected_fields = self.config.expected_fields if self.config.expected_fields is not None else self._generate_expected_fields_and_default_response(self.search_dimensions)[0]
        for field in expected_fields:
            if field in response:
                value = response[field]
                if isinstance(value, dict):
                    # Handle dictionary values (e.g., for negation fields)
                    normalized[field] = {}
                    for key, val in value.items():
                        if isinstance(val, list):
                            # Filter out empty strings and normalize
                            normalized[field][key] = [str(item).strip() for item in val if str(item).strip()]
                        else:
                            # Convert single value to list
                            normalized[field][key] = [str(val).strip()] if str(val).strip() else []
                elif isinstance(value, list):
                    # Handle list values (for non-negation fields)
                    normalized[field] = [str(item).strip() for item in value if str(item).strip()]
                else:
                    # Convert single value to list
                    normalized[field] = [str(value).strip()] if str(value).strip() else []
        return normalized


# Convenience function for backward compatibility
def decompose_query_with_llm(query: str, config: Optional[QueryDecomposerConfig] = None, search_dimensions: Optional[List[SearchDimension]] = None) -> Dict[str, List[str]]:
    """
    Convenience function that maintains the original interface from the notebook.
    
    Args:
        query: The natural language query to decompose
        config: Optional configuration object
        search_dimensions: Optional custom list of SearchDimension objects
        
    Returns:
        Dictionary with decomposed query information
    """
    decomposer = QueryDecomposer(config, search_dimensions=search_dimensions)
    return decomposer.decompose_query(query)


if __name__ == "__main__":
    # Example usage
    decomposer = QueryDecomposer()
    
    # Test query
    test_query = "Looking for a fitness app interface with progress tracking but without calorie counters. Prefer a minimal and light design."
    
    try:
        result = decomposer.decompose_query(test_query)
        print("Decomposed query:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
