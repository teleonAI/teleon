"""
Policy DSL Parser.

Parses YAML/JSON/dict policy definitions into PolicyDefinition models.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from teleon.sentinel.policy_dsl.models import PolicyDefinition, PolicyRule, RuleType


class PolicyParser:
    """Parses policy definitions from YAML, JSON, or dictionaries."""

    @staticmethod
    def parse_file(path: Union[str, Path]) -> List[PolicyDefinition]:
        """Parse a YAML or JSON policy file."""
        path = Path(path)
        content = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML policy files. "
                    "Install it with: pip install pyyaml"
                )
            data = yaml.safe_load(content)
        elif path.suffix == ".json":
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported policy file format: {path.suffix}")

        return PolicyParser._parse_data(data)

    @staticmethod
    def parse_yaml(yaml_str: str) -> List[PolicyDefinition]:
        """Parse a YAML string into policy definitions."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required. Install with: pip install pyyaml"
            )
        data = yaml.safe_load(yaml_str)
        return PolicyParser._parse_data(data)

    @staticmethod
    def parse_dict(data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[PolicyDefinition]:
        """Parse a dictionary or list of dictionaries into policy definitions."""
        if isinstance(data, list):
            return [PolicyParser._parse_single_policy(p) for p in data]
        return PolicyParser._parse_data(data)

    @staticmethod
    def _parse_data(data: Any) -> List[PolicyDefinition]:
        if not isinstance(data, dict):
            raise ValueError("Policy data must be a dictionary with a 'policies' key or a list")

        policies_raw = data.get("policies", [])
        if not isinstance(policies_raw, list):
            raise ValueError("'policies' must be a list")

        return [PolicyParser._parse_single_policy(p) for p in policies_raw]

    @staticmethod
    def _parse_single_policy(raw: Dict[str, Any]) -> PolicyDefinition:
        rules = []
        for rule_raw in raw.get("rules", []):
            rule_type = rule_raw.get("type")
            if isinstance(rule_type, str):
                rule_type = RuleType(rule_type)
            rules.append(PolicyRule(type=rule_type, **{k: v for k, v in rule_raw.items() if k != "type"}))

        return PolicyDefinition(
            name=raw["name"],
            description=raw.get("description"),
            enabled=raw.get("enabled", True),
            severity=raw.get("severity", "medium"),
            action=raw.get("action", "flag"),
            targets=raw.get("targets", ["input", "output"]),
            rules=rules,
            match=raw.get("match", "any"),
            agent_ids=raw.get("agent_ids"),
        )
