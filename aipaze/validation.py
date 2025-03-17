import inspect
from functools import wraps
from typing import Any, Dict, Type, Optional, Callable, get_type_hints
import logging

try:
    from pydantic import BaseModel, Field, ValidationError, create_model
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Define fallback classes
    class BaseModel:
        pass
    class ValidationError(Exception):
        pass

def validate(input_schema=None, output_schema=None):
    """
    Decorator to validate inputs and outputs of resources or tools.
    
    Args:
        input_schema: Pydantic model for input validation
        output_schema: Pydantic model for output validation
        
    Returns:
        Decorated function with validation
    """
    if not PYDANTIC_AVAILABLE:
        logging.warning("Pydantic not installed. Validation will be skipped.")
        
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip validation if Pydantic not available
            if not PYDANTIC_AVAILABLE:
                return func(*args, **kwargs)
                
            # Validate input
            if input_schema:
                try:
                    validated_kwargs = input_schema(**kwargs)
                    # Convert back to dict for function call
                    kwargs = validated_kwargs.model_dump()
                except ValidationError as e:
                    raise ValueError(f"Input validation error: {e}")
                except AttributeError:
                    # For older Pydantic versions
                    kwargs = validated_kwargs.dict()
            
            # Call function
            result = func(*args, **kwargs)
            
            # Validate output
            if output_schema:
                try:
                    if isinstance(result, dict):
                        validated_result = output_schema(**result)
                        # Convert back to dict for return
                        try:
                            return validated_result.model_dump()
                        except AttributeError:
                            # For older Pydantic versions
                            return validated_result.dict()
                    else:
                        raise ValueError(f"Output must be a dict, got {type(result)}")
                except ValidationError as e:
                    raise ValueError(f"Output validation error: {e}")
            
            return result
        return wrapper
    return decorator

def infer_schema(func: Callable) -> Dict[str, Any]:
    """
    Infer a validation schema from function type hints.
    
    Args:
        func: Function to analyze
        
    Returns:
        Dictionary mapping parameter names to types
    """
    if not PYDANTIC_AVAILABLE:
        logging.warning("Pydantic not installed. Schema inference will be limited.")
        return {}
        
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    
    field_definitions = {}
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
            
        param_type = hints.get(param_name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        
        # Create Field with default if available
        if default is not ...:
            field_definitions[param_name] = (param_type, Field(default=default))
        else:
            field_definitions[param_name] = (param_type, ...)
    
    return field_definitions

def create_validator(func: Callable) -> Callable:
    """
    Create a validator for a function based on its type hints.
    
    Args:
        func: Function to create validator for
        
    Returns:
        Decorated function with validation
    """
    if not PYDANTIC_AVAILABLE:
        logging.warning("Pydantic not installed. Validation will be skipped.")
        return func
        
    # Infer schemas
    input_fields = infer_schema(func)
    
    # Create input model if we have type hints
    input_model = None
    if input_fields:
        input_model = create_model(f"{func.__name__}Input", **input_fields)
    
    # Try to get return type hint for output model
    hints = get_type_hints(func)
    output_model = None
    if 'return' in hints and hints['return'] != None:
        return_type = hints['return']
        if hasattr(return_type, '__origin__') and return_type.__origin__ is dict:
            # For dict return types, we'll just do basic dict validation
            pass
        
    # Apply validation
    return validate(input_model, output_model)(func)