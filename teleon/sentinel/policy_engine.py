"""
Policy Engine - Custom Policy Evaluation.

Retained for backward compatibility. New code should use the Policy DSL
(teleon.sentinel.policy_dsl) which provides safe YAML-based evaluation
without eval().

The 'condition' policy type now delegates to SafeEvaluator operators
instead of using eval().
"""

import re
from typing import Dict, Any, List, Optional, Callable
from teleon.core import StructuredLogger, LogLevel


class PolicyEngine:
    """
    Custom policy evaluation engine (backward-compatible).

    Supports:
    - Regex pattern matching
    - Condition evaluation (now via safe operator dispatch, no eval)
    - Custom function evaluation (with timeout)
    """

    def __init__(self, policies: Optional[Dict[str, Dict[str, Any]]] = None):
        self.policies: Dict[str, Dict[str, Any]] = policies or {}
        self.logger = StructuredLogger("policy_engine", LogLevel.INFO)

    def add_policy(self, name: str, policy_def: Dict[str, Any]) -> None:
        """Add a custom policy."""
        self.policies[name] = policy_def
        self.logger.debug("Policy added", policy_name=name)

    def evaluate_policy(self, policy_name: str, data: Any) -> List[Dict[str, Any]]:
        """Evaluate a specific policy."""
        if policy_name not in self.policies:
            self.logger.warning("Policy not found", policy_name=policy_name)
            return []

        policy = self.policies[policy_name]
        violations = []
        text = self._extract_text(data)
        policy_type = policy.get("type", "regex")

        if policy_type == "regex":
            pattern = policy.get("pattern")
            if pattern:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append({
                        "type": "custom_policy",
                        "policy_name": policy_name,
                        "message": policy.get("message", f"Policy {policy_name} violation"),
                        "severity": policy.get("severity", "medium"),
                    })

        elif policy_type == "condition":
            # Safe evaluation via operator dispatch (replaces eval())
            violations.extend(self._evaluate_condition_safely(policy_name, policy, text, data))

        elif policy_type == "function":
            func = policy.get("function")
            timeout = policy.get("timeout", 5.0)
            if func and callable(func):
                try:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, data)
                        result = future.result(timeout=timeout)
                    if result:
                        violations.append({
                            "type": "custom_policy",
                            "policy_name": policy_name,
                            "message": policy.get("message", f"Policy {policy_name} violation"),
                            "severity": policy.get("severity", "medium"),
                            "function_result": result,
                        })
                except Exception as e:
                    self.logger.warning(
                        "Policy function evaluation failed",
                        policy_name=policy_name,
                        error=str(e),
                    )

        return violations

    def _evaluate_condition_safely(
        self, policy_name: str, policy: Dict[str, Any], text: str, data: Any
    ) -> List[Dict[str, Any]]:
        """Evaluate a condition policy using safe operators instead of eval()."""
        condition = policy.get("condition", "")
        if not condition:
            return []

        try:
            from teleon.sentinel.policy_dsl.evaluator import SafeEvaluator
            from teleon.sentinel.policy_dsl.models import (
                EvaluationContext,
                PolicyDefinition,
                PolicyRule,
                RuleType,
            )

            # Try to map legacy condition strings to DSL rules.
            # Common patterns: "len(text) > N", "'word' in text"
            ctx = EvaluationContext(
                text=text,
                data=data if isinstance(data, dict) else {"raw": data},
            )

            # Simple heuristic mapping for common legacy conditions
            evaluator = SafeEvaluator()

            # len(text) > N
            len_match = re.match(r"len\(text\)\s*(>|>=|<|<=|==|!=)\s*(\d+)", condition)
            if len_match:
                op_map = {">": "gt", ">=": "gte", "<": "lt", "<=": "lte", "==": "eq", "!=": "neq"}
                rule = PolicyRule(
                    type=RuleType.LENGTH,
                    operator=op_map.get(len_match.group(1), "gt"),
                    value=int(len_match.group(2)),
                )
                policy_def = PolicyDefinition(
                    name=policy_name,
                    description=policy.get("message"),
                    severity=policy.get("severity", "medium"),
                    rules=[rule],
                )
                return evaluator.evaluate_policy(policy_def, ctx)

            # 'word' in text
            in_match = re.match(r"['\"](.+?)['\"]\s+in\s+text", condition)
            if in_match:
                rule = PolicyRule(
                    type=RuleType.TEXT_MATCH,
                    operator="contains",
                    value=in_match.group(1),
                )
                policy_def = PolicyDefinition(
                    name=policy_name,
                    description=policy.get("message"),
                    severity=policy.get("severity", "medium"),
                    rules=[rule],
                )
                return evaluator.evaluate_policy(policy_def, ctx)

            self.logger.warning(
                "Could not parse legacy condition, skipping",
                policy_name=policy_name,
                condition=condition[:100],
            )
            return []

        except Exception as e:
            self.logger.warning(
                "Condition evaluation failed",
                policy_name=policy_name,
                error=str(e),
            )
            return []

    def evaluate_all(self, data: Any) -> List[Dict[str, Any]]:
        """Evaluate all policies."""
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
            return " ".join(texts)
        elif isinstance(data, list):
            texts = [self._extract_text(item) for item in data]
            return " ".join(texts)
        else:
            return str(data)
