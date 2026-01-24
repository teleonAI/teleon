"""
Policy Engine - Custom Policy Evaluation.

Evaluates custom policies defined by users.
"""

import re
from typing import Dict, Any, List, Optional, Callable
from teleon.core import StructuredLogger, LogLevel


class PolicyEngine:
    """
    Custom policy evaluation engine.
    
    Supports:
    - Regex pattern matching
    - Condition evaluation
    - Custom function evaluation
    """
    
    def __init__(self, policies: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize policy engine.
        
        Args:
            policies: Dictionary of policy definitions
        """
        self.policies: Dict[str, Dict[str, Any]] = policies or {}
        self.logger = StructuredLogger("policy_engine", LogLevel.INFO)
    
    def add_policy(self, name: str, policy_def: Dict[str, Any]) -> None:
        """
        Add a custom policy.
        
        Args:
            name: Policy name
            policy_def: Policy definition with:
                - type: 'regex', 'condition', or 'function'
                - pattern: Regex pattern or condition expression
                - action: Action to take on match
                - message: Violation message
        """
        self.policies[name] = policy_def
        self.logger.debug("Policy added", policy_name=name)
    
    def evaluate_policy(self, policy_name: str, data: Any) -> List[Dict[str, Any]]:
        """
        Evaluate a specific policy.
        
        Args:
            policy_name: Name of policy to evaluate
            data: Data to evaluate
        
        Returns:
            List of violations
        """
        if policy_name not in self.policies:
            self.logger.warning("Policy not found", policy_name=policy_name)
            return []
        
        policy = self.policies[policy_name]
        violations = []
        
        # Extract text from data
        text = self._extract_text(data)
        
        policy_type = policy.get('type', 'regex')
        
        if policy_type == 'regex':
            pattern = policy.get('pattern')
            if pattern:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append({
                        'type': 'custom_policy',
                        'policy_name': policy_name,
                        'message': policy.get('message', f'Policy {policy_name} violation'),
                        'severity': policy.get('severity', 'medium')
                    })
        
        elif policy_type == 'condition':
            condition = policy.get('condition')
            if condition:
                # Validate condition is safe before evaluation
                if not self._is_safe_condition(condition):
                    self.logger.warning(
                        "Unsafe policy condition rejected",
                        policy_name=policy_name,
                        condition=condition[:100]  # Truncate for logging
                    )
                else:
                    try:
                        # Safe evaluation context with restricted builtins
                        safe_dict = {
                            'text': text,
                            'data': data,
                            'len': len,
                            'str': str,
                            'bool': bool,
                            'int': int,
                            'float': float,
                            'abs': abs,
                            'min': min,
                            'max': max,
                            'sum': sum,
                            'any': any,
                            'all': all,
                            'isinstance': isinstance,
                            'True': True,
                            'False': False,
                            'None': None,
                        }
                        result = eval(condition, {"__builtins__": {}}, safe_dict)
                        if result:
                            violations.append({
                                'type': 'custom_policy',
                                'policy_name': policy_name,
                                'message': policy.get('message', f'Policy {policy_name} violation'),
                                'severity': policy.get('severity', 'medium')
                            })
                    except Exception as e:
                        self.logger.warning(
                            "Policy condition evaluation failed",
                            policy_name=policy_name,
                            error=str(e)
                        )
        
        elif policy_type == 'function':
            func = policy.get('function')
            timeout = policy.get('timeout', 5.0)  # Default 5 second timeout
            if func and callable(func):
                try:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, data)
                        result = future.result(timeout=timeout)
                    if result:
                        violations.append({
                            'type': 'custom_policy',
                            'policy_name': policy_name,
                            'message': policy.get('message', f'Policy {policy_name} violation'),
                            'severity': policy.get('severity', 'medium'),
                            'function_result': result
                        })
                except concurrent.futures.TimeoutError:
                    self.logger.warning(
                        "Policy function timed out",
                        policy_name=policy_name,
                        timeout=timeout
                    )
                except Exception as e:
                    self.logger.warning(
                        "Policy function evaluation failed",
                        policy_name=policy_name,
                        error=str(e)
                    )
        
        return violations
    
    def evaluate_all(self, data: Any) -> List[Dict[str, Any]]:
        """
        Evaluate all policies.
        
        Args:
            data: Data to evaluate
        
        Returns:
            List of all violations
        """
        all_violations = []
        
        for policy_name in self.policies:
            violations = self.evaluate_policy(policy_name, data)
            all_violations.extend(violations)
        
        return all_violations
    
    def _extract_text(self, data: Any) -> str:
        """Extract text from various data types."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            texts = []
            for value in data.values():
                texts.append(self._extract_text(value))
            return ' '.join(texts)
        elif isinstance(data, list):
            texts = [self._extract_text(item) for item in data]
            return ' '.join(texts)
        else:
            return str(data)

    def _is_safe_condition(self, condition: str) -> bool:
        """
        Validate that a condition string is safe for evaluation.

        Blocks dangerous patterns that could be used for code injection.

        Args:
            condition: The condition string to validate

        Returns:
            True if the condition is safe, False otherwise
        """
        if not condition or not isinstance(condition, str):
            return False

        # Maximum condition length to prevent DoS
        if len(condition) > 1000:
            return False

        # Dangerous patterns that should never appear in conditions
        dangerous_patterns = [
            '__',           # Dunder methods/attributes
            'import',       # Import statements
            'exec',         # Code execution
            'eval',         # Nested eval
            'compile',      # Code compilation
            'open',         # File operations
            'file',         # File operations
            'input',        # User input
            'getattr',      # Attribute access
            'setattr',      # Attribute modification
            'delattr',      # Attribute deletion
            'globals',      # Global scope access
            'locals',       # Local scope access
            'vars',         # Variable access
            'dir',          # Directory listing
            'type',         # Type introspection
            'class',        # Class definition
            'lambda',       # Lambda functions
            'def ',         # Function definition
            'os.',          # OS module
            'sys.',         # Sys module
            'subprocess',   # Subprocess execution
            'socket',       # Network operations
            'requests',     # HTTP requests
            'urllib',       # URL operations
            'pickle',       # Serialization
            'marshal',      # Serialization
            'shelve',       # Persistence
            'codecs',       # Codec operations
            'builtins',     # Builtins access
            'breakpoint',   # Debugging
        ]

        condition_lower = condition.lower()
        for pattern in dangerous_patterns:
            if pattern in condition_lower:
                return False

        return True

