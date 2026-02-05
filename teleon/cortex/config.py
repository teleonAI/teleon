"""
Memory configuration.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class AutoContextConfig(BaseModel):
    """Configuration for auto-context injection."""

    enabled: bool = Field(default=True, description="Inject context into LLM")
    history_limit: int = Field(default=10, ge=0, description="Last N conversations to include")
    relevant_limit: int = Field(default=5, ge=0, description="Top N semantic matches to include")
    max_tokens: int = Field(default=2000, ge=100, description="Token budget for context")
    filter: Dict[str, Any] = Field(default_factory=dict, description="Additional filter for auto-retrieval")


class LayerConfig(BaseModel):
    """Configuration for a memory layer."""

    scope: List[str] = Field(default_factory=list, description="Scope fields for this layer")


class MemoryConfig(BaseModel):
    """
    Memory configuration.

    Examples:
        # Simple
        memory=True

        # With options
        memory={
            "auto": True,
            "fields": ["customer_id"],
            "scope": ["tenant_id"],
            "name": "shared-memory",
            "layers": {
                "company": {"scope": []},
                "team": {"scope": ["team_id"]}
            }
        }
    """

    # Auto-save behavior
    auto: bool = Field(default=True, description="Auto-save conversations")
    fields: Optional[List[str]] = Field(default=None, description="Only capture these args (None = all)")

    # Multi-tenancy
    scope: List[str] = Field(default_factory=list, description="Mandatory isolation fields")

    # Sharing
    name: Optional[str] = Field(default=None, description="Share across agents (default: agent name)")

    # Hierarchy
    layers: Dict[str, LayerConfig] = Field(default_factory=dict, description="Custom memory layers")

    # Auto-context injection
    auto_context: AutoContextConfig = Field(default_factory=AutoContextConfig, description="Auto-context config")


def parse_memory_config(config: Union[bool, Dict[str, Any], None]) -> Optional[MemoryConfig]:
    """
    Parse memory configuration from various formats.

    Args:
        config: True, False, None, or dict

    Returns:
        MemoryConfig or None if disabled
    """
    if config is None or config is False:
        return None

    if config is True:
        return MemoryConfig()

    if isinstance(config, dict):
        # Handle layers as dicts
        if "layers" in config and isinstance(config["layers"], dict):
            layers = {}
            for layer_name, layer_cfg in config["layers"].items():
                if isinstance(layer_cfg, dict):
                    layers[layer_name] = LayerConfig(**layer_cfg)
                else:
                    layers[layer_name] = layer_cfg
            config = {**config, "layers": layers}

        # Handle auto_context as dict
        if "auto_context" in config and isinstance(config["auto_context"], dict):
            config = {**config, "auto_context": AutoContextConfig(**config["auto_context"])}

        return MemoryConfig(**config)

    raise ValueError(f"Invalid memory config type: {type(config)}")
