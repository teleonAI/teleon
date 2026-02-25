"""
Safe Policy Evaluator.

Evaluates policy rules using an operator dispatch table.
No eval(), no exec(). All operations are whitelisted.
"""

import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from teleon.sentinel.policy_dsl.models import (
    EvaluationContext,
    PolicyDefinition,
    PolicyRule,
    RuleType,
)


# ---------------------------------------------------------------------------
# Operator dispatch table
# ---------------------------------------------------------------------------

def _op_gt(a: Any, b: Any) -> bool:
    return float(a) > float(b)

def _op_gte(a: Any, b: Any) -> bool:
    return float(a) >= float(b)

def _op_lt(a: Any, b: Any) -> bool:
    return float(a) < float(b)

def _op_lte(a: Any, b: Any) -> bool:
    return float(a) <= float(b)

def _op_eq(a: Any, b: Any) -> bool:
    return a == b

def _op_neq(a: Any, b: Any) -> bool:
    return a != b

def _op_in(a: Any, b: Any) -> bool:
    return a in b

def _op_not_in(a: Any, b: Any) -> bool:
    return a not in b

def _op_contains(a: Any, b: Any) -> bool:
    return str(b).lower() in str(a).lower()

def _op_contains_any(a: Any, b: Any) -> bool:
    a_lower = str(a).lower()
    return any(str(v).lower() in a_lower for v in b)

def _op_contains_all(a: Any, b: Any) -> bool:
    a_lower = str(a).lower()
    return all(str(v).lower() in a_lower for v in b)

def _op_not_contains(a: Any, b: Any) -> bool:
    return str(b).lower() not in str(a).lower()

def _op_not_contains_any(a: Any, b: Any) -> bool:
    a_lower = str(a).lower()
    return not any(str(v).lower() in a_lower for v in b)

def _op_starts_with(a: Any, b: Any) -> bool:
    return str(a).lower().startswith(str(b).lower())

def _op_ends_with(a: Any, b: Any) -> bool:
    return str(a).lower().endswith(str(b).lower())

def _op_between(a: Any, b: Any) -> bool:
    if isinstance(b, (list, tuple)) and len(b) == 2:
        return float(b[0]) <= float(a) <= float(b[1])
    return False

def _op_matches(a: Any, b: Any) -> bool:
    return bool(re.search(str(b), str(a), re.I))


_OPERATORS: Dict[str, Any] = {
    "gt": _op_gt,
    "gte": _op_gte,
    "lt": _op_lt,
    "lte": _op_lte,
    "eq": _op_eq,
    "neq": _op_neq,
    "in": _op_in,
    "not_in": _op_not_in,
    "contains": _op_contains,
    "contains_any": _op_contains_any,
    "contains_all": _op_contains_all,
    "not_contains": _op_not_contains,
    "not_contains_any": _op_not_contains_any,
    "starts_with": _op_starts_with,
    "ends_with": _op_ends_with,
    "between": _op_between,
    "matches": _op_matches,
}


def _resolve_field(data: Dict[str, Any], field_path: str) -> Any:
    """Resolve a dot-path field (e.g. 'data.confidence') from a dict."""
    parts = field_path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
        if current is None:
            return None
    return current


class SafeEvaluator:
    """
    Safe policy rule evaluator.

    Evaluates rules using whitelisted operators and field accessors.
    No eval(), no exec(). Supports rate limiting with in-memory sliding windows.
    """

    def __init__(self):
        # Rate limit tracking: key -> list of timestamps
        self._rate_windows: Dict[str, List[float]] = defaultdict(list)

    def evaluate_policy(
        self,
        policy: PolicyDefinition,
        context: EvaluationContext,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a policy against a context.

        Returns a list of violation dicts (empty = passed).
        """
        if not policy.enabled:
            return []

        # Check if this validation_type is targeted by the policy
        if context.validation_type not in policy.targets:
            return []

        # Check agent_ids filter
        if policy.agent_ids and context.agent_name not in policy.agent_ids:
            return []

        rule_results: List[bool] = []
        for rule in policy.rules:
            matched = self._evaluate_rule(rule, context)
            rule_results.append(matched)

        # Determine overall match
        if policy.match == "all":
            triggered = all(rule_results) if rule_results else False
        else:  # "any" (OR)
            triggered = any(rule_results) if rule_results else False

        if triggered:
            return [
                {
                    "type": "custom_policy",
                    "policy_name": policy.name,
                    "message": policy.description or f"Policy '{policy.name}' violation",
                    "severity": policy.severity,
                    "action": policy.action,
                }
            ]
        return []

    def evaluate_policies(
        self,
        policies: List[PolicyDefinition],
        context: EvaluationContext,
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple policies. Returns all violations."""
        violations: List[Dict[str, Any]] = []
        for policy in policies:
            violations.extend(self.evaluate_policy(policy, context))
        return violations

    def _evaluate_rule(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        """Evaluate a single rule against the context."""
        if rule.type == RuleType.TEXT_MATCH:
            return self._eval_text_match(rule, context)
        elif rule.type == RuleType.REGEX:
            return self._eval_regex(rule, context)
        elif rule.type == RuleType.LENGTH:
            return self._eval_length(rule, context)
        elif rule.type == RuleType.CONDITION:
            return self._eval_condition(rule, context)
        elif rule.type == RuleType.TOOL_ALLOWLIST:
            return self._eval_tool_allowlist(rule, context)
        elif rule.type == RuleType.TOOL_BLOCKLIST:
            return self._eval_tool_blocklist(rule, context)
        elif rule.type == RuleType.TOOL_ARGUMENT:
            return self._eval_tool_argument(rule, context)
        elif rule.type == RuleType.RATE_LIMIT:
            return self._eval_rate_limit(rule, context)
        return False

    def _get_field_value(self, rule: PolicyRule, context: EvaluationContext) -> Any:
        """Get the field value from context for condition/length rules."""
        if rule.field:
            # Try context.data first, then context attributes
            val = _resolve_field(context.data, rule.field)
            if val is not None:
                return val
            return getattr(context, rule.field, None)
        return context.text

    def _eval_text_match(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        text = context.text
        if not rule.case_sensitive:
            text = text.lower()

        operator = rule.operator or "contains"
        compare_value = rule.value
        compare_values = rule.values

        if operator in ("contains_any", "not_contains_any", "contains_all") and compare_values:
            op_func = _OPERATORS.get(operator)
            if op_func:
                if not rule.case_sensitive and compare_values:
                    compare_values = [str(v).lower() for v in compare_values]
                return op_func(text, compare_values)

        if compare_value is not None:
            if not rule.case_sensitive:
                compare_value = str(compare_value).lower()
            op_func = _OPERATORS.get(operator)
            if op_func:
                return op_func(text, compare_value)

        return False

    def _eval_regex(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        if not rule.pattern:
            return False
        flags = 0 if rule.case_sensitive else re.I
        return bool(re.search(rule.pattern, context.text, flags))

    def _eval_length(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        value = self._get_field_value(rule, context)
        length = len(str(value)) if value is not None else 0
        operator = rule.operator or "gt"
        op_func = _OPERATORS.get(operator)
        if op_func and rule.value is not None:
            return op_func(length, rule.value)
        return False

    def _eval_condition(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        field_value = self._get_field_value(rule, context)
        if field_value is None:
            return False
        operator = rule.operator or "eq"
        compare = rule.values if rule.values is not None else rule.value
        op_func = _OPERATORS.get(operator)
        if op_func:
            try:
                return op_func(field_value, compare)
            except (TypeError, ValueError):
                return False
        return False

    def _eval_tool_allowlist(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        if context.validation_type != "tool_call" or not context.tool_name:
            return False
        if rule.allowed_tools is None:
            return False
        return context.tool_name not in rule.allowed_tools

    def _eval_tool_blocklist(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        if context.validation_type != "tool_call" or not context.tool_name:
            return False
        if rule.blocked_tools is None:
            return False
        return context.tool_name in rule.blocked_tools

    def _eval_tool_argument(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        if context.validation_type != "tool_call":
            return False
        if rule.tool and context.tool_name != rule.tool:
            return False
        if not rule.argument or not context.tool_args:
            return False
        arg_value = context.tool_args.get(rule.argument)
        if arg_value is None:
            return False

        operator = rule.operator or "contains"
        compare = rule.values if rule.values is not None else rule.value
        op_func = _OPERATORS.get(operator)
        if op_func and compare is not None:
            try:
                return op_func(str(arg_value), compare)
            except (TypeError, ValueError):
                return False
        return False

    def _eval_rate_limit(self, rule: PolicyRule, context: EvaluationContext) -> bool:
        if rule.max_count is None or rule.window_seconds is None:
            return False

        scope = rule.scope or "agent"
        if scope == "agent":
            key = f"rate:{context.agent_name}"
        else:
            key = f"rate:global"

        now = time.time()
        window_start = now - rule.window_seconds

        # Sliding window: remove expired entries
        self._rate_windows[key] = [
            t for t in self._rate_windows[key] if t > window_start
        ]
        self._rate_windows[key].append(now)

        return len(self._rate_windows[key]) > rule.max_count
