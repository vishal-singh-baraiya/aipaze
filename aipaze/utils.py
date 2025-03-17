import logging
import socket
import time
import traceback
from functools import wraps
import asyncio
from typing import Callable, TypeVar, Any

T = TypeVar('T')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def retry(max_attempts: int, delay: int):
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
    """Check if an object can be JSON serialized"""
    try:
        import json
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False