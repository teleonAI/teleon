"""Agent configuration models."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from inspect import Signature


@dataclass
class AgentConfig:
    """
    Configuration for a Teleon agent.
    
    This class holds all configuration options for an agent, including:
    - Basic metadata (name, version)
    - Memory configuration
    - Scaling parameters
    - LLM settings
    - Collaboration settings
    - Resource limits
    """
    
    name: str
    memory: bool = False
    scale: Dict[str, Any] = field(default_factory=lambda: {'min': 1, 'max': 10})
    llm: Dict[str, Any] = field(default_factory=dict)
    tools: list = field(default_factory=list)
    collaborate: bool = False
    timeout: Optional[float] = None
    signature: Optional[Signature] = None
    version: str = "1.0.0"
    
    # Additional configuration
    resources: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    secrets: list = field(default_factory=list)
    retry: Dict[str, Any] = field(default_factory=dict)
    circuit_breaker: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """
        Validate configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate name
        if not self.name:
            raise ValueError("Agent name cannot be empty")
        
        if not self.name.replace('-', '').replace('_', '').isalnum():
            raise ValueError(
                f"Agent name '{self.name}' must contain only alphanumeric "
                "characters, hyphens, and underscores"
            )
        
        # Validate scale configuration
        if self.scale:
            min_replicas = self.scale.get('min', 1)
            max_replicas = self.scale.get('max', 10)
            
            if min_replicas < 0:
                raise ValueError("scale.min must be >= 0")
            
            if max_replicas < min_replicas:
                raise ValueError("scale.max must be >= scale.min")
            
            if max_replicas > 1000:
                raise ValueError("scale.max cannot exceed 1000")
        
        # Validate timeout
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = asdict(self)
        # Remove signature as it's not serializable
        data.pop('signature', None)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create configuration from dictionary."""
        return cls(**data)

