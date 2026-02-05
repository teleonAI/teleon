"""
Scope enforcement for multi-tenancy.
"""

from typing import Any, Dict, List
import logging

logger = logging.getLogger("teleon.cortex.scope")


class ScopeEnforcer:
    """
    Enforces scope fields on all memory operations.
    Prevents accidental cross-tenant data access.
    """

    def __init__(self, scope_fields: List[str], scope_values: Dict[str, Any]):
        """
        Initialize scope enforcer.

        Args:
            scope_fields: List of field names that must be enforced
            scope_values: Values for each scope field (extracted from function args)
        """
        self._scope_fields = scope_fields
        self._scope_values = scope_values

        # Validate that all scope fields have values
        missing = [f for f in scope_fields if f not in scope_values]
        if missing:
            logger.warning(f"Scope fields missing values: {missing}")

    def enforce_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add scope fields to store fields.
        Scope values take precedence (cannot be overridden by user).
        """
        result = dict(fields)
        for field in self._scope_fields:
            if field in self._scope_values:
                if field in result and result[field] != self._scope_values[field]:
                    logger.warning(
                        f"Scope field '{field}' override attempted: "
                        f"'{result[field]}' → '{self._scope_values[field]}'"
                    )
                result[field] = self._scope_values[field]
        return result

    def enforce_filter(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add scope fields to query filter.
        Scope values take precedence (cannot be overridden by user).
        """
        result = dict(filter)
        for field in self._scope_fields:
            if field in self._scope_values:
                if field in result and result[field] != self._scope_values[field]:
                    logger.warning(
                        f"Scope field '{field}' filter override attempted: "
                        f"'{result[field]}' → '{self._scope_values[field]}'"
                    )
                result[field] = self._scope_values[field]
        return result
