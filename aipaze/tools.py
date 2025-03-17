import inspect
import json
import logging
import asyncio
from typing import Callable, Dict, List, Any, Optional

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, func: Callable, description: str = None):
        """Register a tool with the registry"""
        if description is None:
            description = func.__doc__ or f"Tool for {name}"
        else:
            # If both description and docstring are provided, use description
            # and update the function's docstring to match
            if func.__doc__ is None:
                func.__doc__ = description
        
        # Get function signature for parameters
        sig = inspect.signature(func)
        parameters = {}
        for param_name, param in sig.parameters.items():
            # Skip self parameter for methods
            if param_name == 'self':
                continue
            
            param_info = {
                "type": "string",  # Default to string
                "description": f"Parameter {param_name} for {name}"
            }
            
            # Try to get type annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == str:
                    param_info["type"] = "string"
                elif param.annotation == int:
                    param_info["type"] = "integer"
                elif param.annotation == float:
                    param_info["type"] = "number"
                elif param.annotation == bool:
                    param_info["type"] = "boolean"
                # Add more type mappings as needed
            
            parameters[param_name] = param_info
        
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
        
        logging.info(f"Registered tool: {name} with description: {description[:50]}...")
        return func
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name"""
        if name in self.tools:
            return self.tools[name]["function"]
        return None
    
    def get_tool_specs(self) -> List[Dict[str, Any]]:
        """Get tool specifications in a format suitable for LLMs"""
        specs = []
        for name, tool_info in self.tools.items():
            spec = {
                "name": name,
                "description": tool_info["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool_info["parameters"],
                    "required": list(tool_info["parameters"].keys())
                }
            }
            specs.append(spec)
        return specs
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with the given arguments"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        
        func = self.tools[name]["function"]
        logging.info(f"Executing tool: {name} with args: {kwargs}")
        result = func(**kwargs)
        
        # Handle coroutines if necessary
        if asyncio.iscoroutine(result):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.ensure_future(result)
            else:
                return asyncio.run(result)
        
        return result

# Global registry instance
_registry = ToolRegistry()

def tool(name: str = None, description: str = None):
    """
    Decorator to register a tool.
    
    Args:
        name: The name of the tool (optional, defaults to function name)
        description: A description of what the tool does and when to use it
    """
    def decorator(func):
        nonlocal name
        if name is None:
            name = func.__name__
        _registry.register(name, func, description)
        return func
    return decorator

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry"""
    return _registry