from typing import Any, Callable
import json
import logging

# Try to import optional dependencies
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

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
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package is not installed. Install with: pip install openai")
        
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

def connect_llm_local(model_path: str, device: str = "cpu") -> Callable:
    """
    Connect to a local LLM using transformers library
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers package is not installed. Install with: pip install transformers torch")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map=device,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    
    def llm_call(input_data: Any) -> str:
        try:
            # Convert to messages format if needed
            if isinstance(input_data, dict) and "messages" in input_data:
                if hasattr(tokenizer, "apply_chat_template"):
                    prompt = tokenizer.apply_chat_template(
                        input_data["messages"], 
                        tokenize=False
                    )
                else:
                    # Fallback for tokenizers without chat template
                    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in input_data["messages"]])
            else:
                prompt = str(input_data)
                
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                inputs.input_ids, 
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7
            )
            return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Local LLM error: {str(e)}")
            raise
    
    return llm_call

def connect_multimodal_llm(model: str, api_key: str, base_url: str = None) -> Callable:
    """
    Connect to a multimodal LLM that supports images
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package is not installed. Install with: pip install openai")
        
    if not api_key:
        raise ValueError("API key is required")
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else "https://api.openai.com/v1"
    )
    
    def llm_call(input_data: Any) -> str:
        try:
            messages = []
            
            if isinstance(input_data, dict) and "messages" in input_data:
                # Process messages that may contain image URLs
                for msg in input_data["messages"]:
                    if "image_url" in msg:
                        messages.append({
                            "role": msg.get("role", "user"),
                            "content": [
                                {"type": "text", "text": msg.get("content", "")},
                                {"type": "image_url", "image_url": msg["image_url"]}
                            ]
                        })
                    else:
                        messages.append(msg)
            else:
                messages = [{"role": "user", "content": str(input_data)}]
                
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Multimodal LLM error: {str(e)}")
            raise
    
    return llm_call