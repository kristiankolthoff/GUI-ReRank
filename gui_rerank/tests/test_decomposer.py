#!/usr/bin/env python3
"""
Test script for the QueryDecomposer class.
Demonstrates different configuration options and usage patterns.
"""

import json
import os
from query_decomposition import QueryDecomposer, QueryDecomposerConfig, decompose_query_with_llm
from llm.llm import LLM


def test_basic_usage():
    """Test basic usage of QueryDecomposer."""
    print("=== Testing Basic Usage ===")
    
    decomposer = QueryDecomposer()
    query = "Looking for a fitness app interface with progress tracking but without calorie counters. Prefer a minimal and light design."
    
    try:
        result = decomposer.decompose_query(query)
        print("Query:", query)
        print("Result:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")


def test_custom_config():
    """Test QueryDecomposer with custom configuration."""
    print("\n=== Testing Custom Configuration ===")
    
    # Create custom LLM
    custom_llm = LLM(
        model_name=LLM.MODEL_GPT_4O,
        temperature=0.1,  # Lower temperature for more consistent results
        max_tokens=256,   # Smaller response
    )
    
    # Create custom configuration
    config = QueryDecomposerConfig(
        llm=custom_llm,
        raise_on_json_error=False,  # Don't raise on JSON errors
    )
    
    decomposer = QueryDecomposer(config)
    query = "A login screen without social media buttons, modern and clean style, no dark background, from a finance app"
    
    try:
        result = decomposer.decompose_query(query)
        print("Query:", query)
        print("Result:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")


def test_different_llm_models():
    """Test QueryDecomposer with different LLM models."""
    print("\n=== Testing Different LLM Models ===")
    
    query = "Looking for period setup screen to change period length and cycle length but without start date selection of last period. Prefer a minimal and light design."
    
    # Test with different models
    models_to_test = [
        LLM.MODEL_GPT_4O,
        LLM.MODEL_GEMINI_2_0_FLASH,
        LLM.MODEL_CLAUDE_SONNET_3_7
    ]
    
    for model_name in models_to_test:
        try:
            print(f"\nTesting with model: {model_name}")
            llm = LLM(model_name=model_name, temperature=0.2, max_tokens=512)
            config = QueryDecomposerConfig(llm=llm)
            decomposer = QueryDecomposer(config)
            
            result = decomposer.decompose_query(query)
            print("Result:")
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")


def test_backward_compatibility():
    """Test the backward compatibility function."""
    print("\n=== Testing Backward Compatibility ===")
    
    query = "Looking for a fitness app interface with progress tracking but without calorie counters. Prefer a minimal and light design."
    
    try:
        result = decompose_query_with_llm(query)
        print("Query:", query)
        print("Result:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")


def test_error_handling():
    """Test error handling with invalid queries."""
    print("\n=== Testing Error Handling ===")
    
    # Test with empty query
    decomposer = QueryDecomposer()
    
    try:
        result = decomposer.decompose_query("")
        print("Empty query result:", result)
    except ValueError as e:
        print(f"Expected error for empty query: {e}")
    
    # Test with very short query
    try:
        result = decomposer.decompose_query("test")
        print("Short query result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error with short query: {e}")


def test_custom_prompt_template():
    """Test with custom prompt template path."""
    print("\n=== Testing Custom Prompt Template ===")
    
    config = QueryDecomposerConfig(
        prompt_template_path="data/prompts/query_decomposition.txt"
    )
    decomposer = QueryDecomposer(config)
    
    query = "A workout app with timer functionality, clean white design, not a meditation app"
    
    try:
        result = decomposer.decompose_query(query)
        print("Query:", query)
        print("Result:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Check if required environment variables are set
    required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Warning: Missing environment variables: {missing_vars}")
        print("Some tests may fail. Please set the required API keys.")
    
    # Run all tests
    test_basic_usage()
    test_custom_config()
    test_different_llm_models()
    test_backward_compatibility()
    test_error_handling()
    test_custom_prompt_template()
    
    print("\n=== All tests completed ===") 