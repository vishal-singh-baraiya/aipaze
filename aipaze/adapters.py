from typing import Any, Callable
import json
import logging
from openai import OpenAI

def adapt_llm(llm: Callable) -> Callable:
    """Normalize LLM input/output for consistency."""
    def wrapper(input_data):
        try:
            # Handle different input formats
            if isinstance(input_data, dict):
                # Check if it's a messages format
                if "messages" in input_data:
                    # This is already in the right format for most LLMs
                    input_for_llm = input_data
                else:
                    # Convert to JSON string
                    input_for_llm = json.dumps(input_data)
            else:
                # Convert to string
                input_for_llm = str(input_data)
            
            # Call the LLM
            result = llm(input_for_llm)
            
            # Process the result
            if isinstance(result, str):
                # Try to parse as JSON
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return result
            else:
                return result
        except Exception as e:
            logging.error(f"LLM adapter error: {str(e)}")
            return {"error": str(e)}
    return wrapper

def connect_llm_openai(model: str, api_key: str, base_url: str = None) -> Callable:
    """
    Create an OpenAI-compatible LLM client using model, API key, and optional base URL.
    """
    if not api_key:
        raise ValueError("API key is required")
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else "https://api.openai.com/v1"  # Default to OpenAI
    )
    
    def llm_call(input_data: Any) -> str:
        try:
            # Check if input is already in messages format
            if isinstance(input_data, dict) and "messages" in input_data:
                messages = input_data["messages"]
            else:
                # Default to simple user message
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that returns structured output when possible."},
                    {"role": "user", "content": str(input_data)}
                ]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API error: {str(e)}")
            raise
    
    return llm_call