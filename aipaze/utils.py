import logging
import socket
import time
import traceback
from functools import wraps
import asyncio
from typing import Callable, TypeVar, Any, Dict, List

T = TypeVar('T')

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

def get_free_port() -> int:
    """
    Get a free port on the local machine.
    
    Returns:
        Available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def retry(max_attempts: int, delay: int):
    """
    Decorator for retrying async functions.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logging.error(f"All {max_attempts} attempts failed. Last error: {str(e)}")
                        logging.error(traceback.format_exc())
                        raise e
                    logging.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
            # This should never be reached due to the raise in the loop
            raise last_exception if last_exception else RuntimeError("Unknown error in retry")
        return wrapper
    return decorator

def is_json_serializable(obj: Any) -> bool:
    """
    Check if an object can be JSON serialized.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object can be JSON serialized
    """
    try:
        import json
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Recursively merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (overrides dict1)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def timed(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def async_timed(func):
    """
    Decorator to time async function execution.
    
    Args:
        func: Async function to time
        
    Returns:
        Decorated async function
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Async function {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper