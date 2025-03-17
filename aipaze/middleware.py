import logging
from typing import Callable, List, Any, Dict, Optional
from functools import wraps

# Global registry for middleware
_middleware_registry: List[Callable] = []

def middleware(func):
    """
    Decorator to register a global middleware function.
    
    A middleware function should have the signature:
    async def middleware_func(resource, args, next_handler)
    
    Args:
        func: The middleware function
        
    Returns:
        The middleware function
    """
    _middleware_registry.append(func)
    return func

def get_middleware_registry() -> List[Callable]:
    """Get the global middleware registry."""
    return _middleware_registry

async def apply_middleware(resource: str, args: List[Any], result: Any, middleware_list: List[Callable]) -> Any:
    """
    Apply middleware functions to a result.
    
    Args:
        resource: Resource name
        args: Resource arguments
        result: Result to process
        middleware_list: List of middleware functions
        
    Returns:
        Processed result
    """
    processed_result = result
    for middleware_func in middleware_list:
        try:
            processed_result = await middleware_func(resource, args, processed_result)
        except Exception as e:
            logging.error(f"Error in middleware {middleware_func.__name__}: {str(e)}")
    return processed_result

def use_middleware(client, middleware_func, resources=None):
    """
    Apply middleware to a client.
    
    Args:
        client: MCPClient instance
        middleware_func: Middleware function
        resources: Optional list of resources to apply middleware to
        
    Returns:
        The client instance for chaining
    """
    client.middleware.append((middleware_func, resources))
    return client

# Common middleware implementations
@middleware
async def logging_middleware(resource, args, result):
    """Log all resource calls and results."""
    logging.info(f"Resource: {resource}, Args: {args}, Result: {result}")
    return result

@middleware
async def caching_middleware(resource, args, result):
    """Simple in-memory cache for resource results."""
    # This would be expanded in a real implementation
    cache_key = f"{resource}:{str(args)}"
    # Here you would check a cache and return cached result if available
    return result