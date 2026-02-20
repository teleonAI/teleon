"""Input/output validation for Teleon agents."""

from typing import Any, Dict, get_type_hints, get_args, get_origin
from inspect import signature, Parameter
from pydantic import BaseModel, create_model, ValidationError


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class SchemaGenerator:
    """Generate validation schemas from function signatures."""
    
    @staticmethod
    def from_function(func: Any) -> Dict[str, Any]:
        """
        Generate JSON schema from function signature.
        
        Args:
            func: Function to generate schema for
        
        Returns:
            JSON schema dictionary
        """
        sig = signature(func)
        type_hints = get_type_hints(func)
        
        properties = {}
        required = []

        excluded_params = {"self", "cortex"}
        
        for param_name, param in sig.parameters.items():
            if param_name in excluded_params:
                continue
            
            param_type = type_hints.get(param_name, Any)
            param_schema = SchemaGenerator._type_to_schema(param_type)
            
            properties[param_name] = param_schema
            
            # Check if required (no default value)
            if param.default == Parameter.empty:
                required.append(param_name)
        
        schema = {
            'type': 'object',
            'properties': properties,
        }
        
        if required:
            schema['required'] = required
        
        return schema
    
    @staticmethod
    def _type_to_schema(type_hint: Any) -> Dict[str, Any]:
        """Convert Python type hint to JSON schema."""
        origin = get_origin(type_hint)
        
        # Handle Optional[T]
        if origin is type(None) or type_hint is type(None):
            return {'type': 'null'}
        
        # Handle Union types (including Optional)
        if origin is type(int) or type_hint is int:
            return {'type': 'integer'}
        elif origin is type(float) or type_hint is float:
            return {'type': 'number'}
        elif origin is type(str) or type_hint is str:
            return {'type': 'string'}
        elif origin is type(bool) or type_hint is bool:
            return {'type': 'boolean'}
        elif origin is list:
            args = get_args(type_hint)
            item_schema = SchemaGenerator._type_to_schema(args[0]) if args else {'type': 'any'}
            return {'type': 'array', 'items': item_schema}
        elif origin is dict:
            return {'type': 'object'}
        else:
            # Default to any
            return {'type': 'any'}


class Validator:
    """Validate inputs and outputs against schemas."""
    
    @staticmethod
    def validate_input(func: Any, **kwargs: Any) -> None:
        """
        Validate function input arguments.
        
        Args:
            func: Function to validate against
            **kwargs: Arguments to validate
        
        Raises:
            ValidationError: If validation fails
        """
        sig = signature(func)
        type_hints = get_type_hints(func)

        excluded_params = {"self", "cortex"}
        
        # Check for missing required parameters
        for param_name, param in sig.parameters.items():
            if param_name in excluded_params:
                continue
            
            if param.default == Parameter.empty and param_name not in kwargs:
                raise ValidationError(f"Missing required parameter: {param_name}")
        
        # Validate types
        for param_name, value in kwargs.items():
            if param_name not in type_hints:
                continue
            
            expected_type = type_hints[param_name]
            
            # Skip validation for Any type
            if expected_type is Any:
                continue
            
            # Basic type checking
            if not Validator._check_type(value, expected_type):
                raise ValidationError(
                    f"Parameter '{param_name}' expected type {expected_type}, "
                    f"got {type(value)}"
                )
    
    @staticmethod
    def validate_output(func: Any, output: Any) -> None:
        """
        Validate function output.
        
        Args:
            func: Function to validate against
            output: Output value to validate
        
        Raises:
            ValidationError: If validation fails
        """
        type_hints = get_type_hints(func)
        
        if 'return' not in type_hints:
            return
        
        expected_type = type_hints['return']
        
        # Skip validation for Any type
        if expected_type is Any:
            return
        
        if not Validator._check_type(output, expected_type):
            raise ValidationError(
                f"Return value expected type {expected_type}, got {type(output)}"
            )
    
    @staticmethod
    def _check_type(value: Any, expected_type: Any) -> bool:
        """Check if value matches expected type."""
        origin = get_origin(expected_type)
        
        # Handle None
        if expected_type is type(None):
            return value is None
        
        # Handle basic types
        if origin is None:
            return isinstance(value, expected_type)
        
        # Handle List
        if origin is list:
            if not isinstance(value, list):
                return False
            args = get_args(expected_type)
            if args:
                return all(Validator._check_type(item, args[0]) for item in value)
            return True
        
        # Handle Dict
        if origin is dict:
            return isinstance(value, dict)
        
        # Default to True for complex types
        return True


def validate_agent_input(func: Any) -> Any:
    """
    Decorator to add input validation to agent functions.
    
    Example:
        @validate_agent_input
        async def my_agent(x: int, y: str) -> dict:
            return {'result': x}
    """
    def decorator(*args: Any, **kwargs: Any) -> Any:
        Validator.validate_input(func, **kwargs)
        return func(*args, **kwargs)
    
    return decorator

