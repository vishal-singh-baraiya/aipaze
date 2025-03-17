from typing import Callable, Any, Dict
import asyncio
import logging
import inspect

class ResourceManager:
    def __init__(self):
        self.resources: Dict[str, Callable] = {}
        self.dependencies: Dict[str, list] = {}

    def register(self, name: str, func: Callable):
        self.resources[name] = func
        self.dependencies[name] = []  # Add dependency tracking if needed
        logging.info(f"Registered resource: {name}")

    async def get_callable(self, name: str) -> Callable:
        if name not in self.resources:
            raise ValueError(f"Resource '{name}' not found")
            
        async def wrapper(*args, **kwargs):
            try:
                func = self.resources[name]
                result = func(*args, **kwargs)
                
                # Handle both synchronous and asynchronous functions
                if asyncio.iscoroutine(result):
                    result = await result
                    
                # No handling for generators here to avoid the syntax error
                return result
            except Exception as e:
                logging.error(f"Error executing resource {name}: {str(e)}")
                raise
        return wrapper

    async def chain(self, *steps: tuple[str, list]) -> Any:
        """Chain resources: pass output of one as input to the next."""
        result = None
        for resource, args in steps:
            if result is not None:
                if isinstance(args, list):
                    args = [result] + args
                else:
                    kwargs = {"input": result}
                    
            callable_func = await self.get_callable(resource)
            result = await callable_func(*args)
        return result
        
    def get_resource_info(self) -> Dict[str, Dict]:
        """Get information about registered resources"""
        info = {}
        for name, func in self.resources.items():
            signature = inspect.signature(func)
            params = [param.name for param in signature.parameters.values()]
            info[name] = {
                "parameters": params,
                "doc": func.__doc__ or "No documentation available"
            }
        return info