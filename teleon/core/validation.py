"""
Production-grade validation system for Teleon.

Features:
- Comprehensive input/output validation
- Schema generation and validation
- Type safety with runtime checks
- Sanitization and normalization
- Security validation (SQL injection, XSS, etc.)
- Performance validation (size limits, complexity)
"""

from typing import Any, Dict, List, Optional, Type, get_args, get_origin, Union
from pydantic import BaseModel, validator, ValidationError
import re
import html
from datetime import datetime
import inspect

from teleon.core.exceptions import AgentValidationError, ToolValidationError


class ValidationRule(BaseModel):
    """Base validation rule."""
    
    name: str
    enabled: bool = True
    error_message: Optional[str] = None


class StringValidation(ValidationRule):
    """String validation rules."""
    
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allow_empty: bool = True
    strip_whitespace: bool = True
    lowercase: bool = False
    uppercase: bool = False


class NumericValidation(ValidationRule):
    """Numeric validation rules."""
    
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allow_inf: bool = False
    allow_nan: bool = False


class CollectionValidation(ValidationRule):
    """Collection (list/dict) validation rules."""
    
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    unique_items: bool = False


class SecurityValidator:
    """Security validation to prevent common attacks."""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bSELECT\b.*\bFROM\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(;\s*DROP\b)",
        r"(--\s*$)",
        r"(/\*.*\*/)",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
    ]
    
    # Command injection patterns
    CMD_INJECTION_PATTERNS = [
        r"[;&|`$()]",
        r"\.\./",
        r"~",
    ]
    
    @classmethod
    def validate_sql_injection(cls, value: str, strict: bool = True) -> bool:
        """
        Check for SQL injection attempts.
        
        Args:
            value: String to validate
            strict: If True, raise on detection; if False, return boolean
        
        Returns:
            True if safe, False if injection detected (strict=False)
        
        Raises:
            AgentValidationError: If SQL injection detected (strict=True)
        """
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                if strict:
                    raise AgentValidationError(
                        "Potential SQL injection detected",
                        {"pattern": pattern, "value": value[:100]}
                    )
                return False
        return True
    
    @classmethod
    def validate_xss(cls, value: str, strict: bool = True) -> bool:
        """
        Check for XSS attempts.
        
        Args:
            value: String to validate
            strict: If True, raise on detection; if False, return boolean
        
        Returns:
            True if safe, False if XSS detected (strict=False)
        
        Raises:
            AgentValidationError: If XSS detected (strict=True)
        """
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                if strict:
                    raise AgentValidationError(
                        "Potential XSS detected",
                        {"pattern": pattern, "value": value[:100]}
                    )
                return False
        return True
    
    @classmethod
    def validate_command_injection(cls, value: str, strict: bool = True) -> bool:
        """
        Check for command injection attempts.
        
        Args:
            value: String to validate
            strict: If True, raise on detection; if False, return boolean
        
        Returns:
            True if safe, False if injection detected (strict=False)
        
        Raises:
            AgentValidationError: If command injection detected (strict=True)
        """
        for pattern in cls.CMD_INJECTION_PATTERNS:
            if re.search(pattern, value):
                if strict:
                    raise AgentValidationError(
                        "Potential command injection detected",
                        {"pattern": pattern, "value": value[:100]}
                    )
                return False
        return True
    
    @classmethod
    def sanitize_string(cls, value: str, allow_html: bool = False) -> str:
        """
        Sanitize string for safe use.
        
        Args:
            value: String to sanitize
            allow_html: If False, escape HTML
        
        Returns:
            Sanitized string
        """
        if not allow_html:
            value = html.escape(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Normalize whitespace
        value = ' '.join(value.split())
        
        return value


class InputValidator:
    """Validates input data against schemas and rules."""
    
    @staticmethod
    def validate_type(value: Any, expected_type: Type) -> bool:
        """
        Validate value matches expected type.
        
        Args:
            value: Value to validate
            expected_type: Expected type
        
        Returns:
            True if valid
        
        Raises:
            AgentValidationError: If type mismatch
        """
        origin = get_origin(expected_type)
        
        # Handle Optional types
        if origin is Union:
            args = get_args(expected_type)
            if type(None) in args:
                if value is None:
                    return True
                # Try other types in Union
                for arg in args:
                    if arg is not type(None):
                        try:
                            if isinstance(value, arg):
                                return True
                        except TypeError:
                            continue
        
        # Handle List types
        elif origin is list:
            if not isinstance(value, list):
                raise AgentValidationError(
                    f"Expected list, got {type(value).__name__}"
                )
            return True
        
        # Handle Dict types
        elif origin is dict:
            if not isinstance(value, dict):
                raise AgentValidationError(
                    f"Expected dict, got {type(value).__name__}"
                )
            return True
        
        # Simple type check
        else:
            if not isinstance(value, expected_type):
                raise AgentValidationError(
                    f"Expected {expected_type.__name__}, got {type(value).__name__}"
                )
        
        return True
    
    @staticmethod
    def validate_string(
        value: str,
        rules: Optional[StringValidation] = None,
        security_check: bool = True
    ) -> str:
        """
        Validate and normalize string.
        
        Args:
            value: String to validate
            rules: Validation rules
            security_check: Perform security checks
        
        Returns:
            Validated/normalized string
        
        Raises:
            AgentValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise AgentValidationError(f"Expected string, got {type(value).__name__}")
        
        if rules:
            # Length validation
            if not rules.allow_empty and len(value) == 0:
                raise AgentValidationError("Empty string not allowed")
            
            if rules.min_length and len(value) < rules.min_length:
                raise AgentValidationError(
                    f"String too short (min: {rules.min_length}, got: {len(value)})"
                )
            
            if rules.max_length and len(value) > rules.max_length:
                raise AgentValidationError(
                    f"String too long (max: {rules.max_length}, got: {len(value)})"
                )
            
            # Pattern validation
            if rules.pattern and not re.match(rules.pattern, value):
                raise AgentValidationError(
                    f"String does not match pattern: {rules.pattern}"
                )
            
            # Normalization
            if rules.strip_whitespace:
                value = value.strip()
            
            if rules.lowercase:
                value = value.lower()
            
            if rules.uppercase:
                value = value.upper()
        
        # Security validation
        if security_check:
            SecurityValidator.validate_sql_injection(value)
            SecurityValidator.validate_xss(value)
            SecurityValidator.validate_command_injection(value)
        
        return value
    
    @staticmethod
    def validate_numeric(
        value: Union[int, float],
        rules: Optional[NumericValidation] = None
    ) -> Union[int, float]:
        """
        Validate numeric value.
        
        Args:
            value: Numeric value
            rules: Validation rules
        
        Returns:
            Validated value
        
        Raises:
            AgentValidationError: If validation fails
        """
        if not isinstance(value, (int, float)):
            raise AgentValidationError(f"Expected number, got {type(value).__name__}")
        
        if rules:
            # Check for inf/nan
            if isinstance(value, float):
                import math
                if math.isinf(value) and not rules.allow_inf:
                    raise AgentValidationError("Infinity not allowed")
                if math.isnan(value) and not rules.allow_nan:
                    raise AgentValidationError("NaN not allowed")
            
            # Range validation
            if rules.min_value is not None and value < rules.min_value:
                raise AgentValidationError(
                    f"Value too small (min: {rules.min_value}, got: {value})"
                )
            
            if rules.max_value is not None and value > rules.max_value:
                raise AgentValidationError(
                    f"Value too large (max: {rules.max_value}, got: {value})"
                )
        
        return value
    
    @staticmethod
    def validate_collection(
        value: Union[List, Dict],
        rules: Optional[CollectionValidation] = None
    ) -> Union[List, Dict]:
        """
        Validate collection (list/dict).
        
        Args:
            value: Collection
            rules: Validation rules
        
        Returns:
            Validated collection
        
        Raises:
            AgentValidationError: If validation fails
        """
        if not isinstance(value, (list, dict)):
            raise AgentValidationError(f"Expected collection, got {type(value).__name__}")
        
        if rules:
            size = len(value)
            
            # Size validation
            if rules.min_items and size < rules.min_items:
                raise AgentValidationError(
                    f"Collection too small (min: {rules.min_items}, got: {size})"
                )
            
            if rules.max_items and size > rules.max_items:
                raise AgentValidationError(
                    f"Collection too large (max: {rules.max_items}, got: {size})"
                )
            
            # Uniqueness validation (lists only)
            if isinstance(value, list) and rules.unique_items:
                if len(value) != len(set(map(str, value))):
                    raise AgentValidationError("Collection must contain unique items")
        
        return value


class SchemaValidator:
    """Validates data against Pydantic schemas."""
    
    @staticmethod
    def validate_against_model(
        data: Dict[str, Any],
        model: Type[BaseModel]
    ) -> BaseModel:
        """
        Validate data against Pydantic model.
        
        Args:
            data: Data to validate
            model: Pydantic model class
        
        Returns:
            Validated model instance
        
        Raises:
            AgentValidationError: If validation fails
        """
        try:
            return model(**data)
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field = " -> ".join(str(x) for x in error["loc"])
                errors.append(f"{field}: {error['msg']}")
            
            raise AgentValidationError(
                "Schema validation failed",
                {"errors": errors, "data": data}
            )
    
    @staticmethod
    def generate_schema_from_function(func: callable) -> Dict[str, Any]:
        """
        Generate JSON schema from function signature.
        
        Args:
            func: Function to analyze
        
        Returns:
            JSON schema
        """
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ["self", "cls"]:
                continue
            
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            
            # Convert Python types to JSON schema types
            if param_type == str:
                prop_schema = {"type": "string"}
            elif param_type == int:
                prop_schema = {"type": "integer"}
            elif param_type == float:
                prop_schema = {"type": "number"}
            elif param_type == bool:
                prop_schema = {"type": "boolean"}
            elif get_origin(param_type) == list:
                prop_schema = {"type": "array"}
            elif get_origin(param_type) == dict:
                prop_schema = {"type": "object"}
            else:
                prop_schema = {"type": "any"}
            
            properties[param_name] = prop_schema
            
            # Check if required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

