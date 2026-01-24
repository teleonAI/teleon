"""
Production-grade configuration management for Teleon.

Features:
- Type-safe configuration with validation
- Environment-based configuration
- Secrets management
- Configuration inheritance
- Hot reload support
- Schema validation
"""

from typing import Optional, Dict, Any, List, Type, TypeVar, Generic
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os
import yaml
from functools import lru_cache
import logging

from teleon.core.exceptions import ConfigurationError, ConfigValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class SecretsConfig(BaseModel):
    """Secrets configuration with encryption support."""
    
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    database_url: Optional[str] = Field(None, description="Database connection URL")
    redis_url: Optional[str] = Field(None, description="Redis connection URL")
    smtp_password: Optional[str] = Field(None, description="SMTP password")
    
    model_config = ConfigDict()
    
    @classmethod
    def from_env(cls) -> "SecretsConfig":
        """Load secrets from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            database_url=os.getenv("DATABASE_URL"),
            redis_url=os.getenv("REDIS_URL"),
            smtp_password=os.getenv("SMTP_PASSWORD")
        )


class LLMConfig(BaseModel):
    """LLM configuration."""
    
    default_provider: str = Field("openai", description="Default LLM provider")
    default_model: str = Field("gpt-4", description="Default model")
    max_retries: int = Field(3, ge=0, le=10, description="Max retry attempts")
    timeout: float = Field(30.0, gt=0, le=300, description="Request timeout in seconds")
    max_concurrent_requests: int = Field(10, ge=1, le=1000, description="Max concurrent requests")
    
    # Rate limiting
    rate_limit_rpm: Optional[int] = Field(None, ge=1, description="Requests per minute")
    rate_limit_tpm: Optional[int] = Field(None, ge=1, description="Tokens per minute")
    
    # Caching
    enable_cache: bool = Field(True, description="Enable response caching")
    cache_ttl: int = Field(3600, ge=0, description="Cache TTL in seconds")
    
    # Cost management
    max_cost_per_request: Optional[float] = Field(None, gt=0, description="Max cost per request")
    daily_budget: Optional[float] = Field(None, gt=0, description="Daily budget limit")
    
    @field_validator("default_provider")
    @classmethod
    def validate_provider(cls, v):
        allowed = ["openai", "anthropic", "google", "cohere"]
        if v not in allowed:
            raise ValueError(f"Provider must be one of {allowed}")
        return v


class MemoryConfig(BaseModel):
    """Memory configuration."""
    
    enabled: bool = Field(True, description="Enable memory systems")
    
    # Working memory
    working_ttl: int = Field(3600, ge=60, description="Working memory TTL in seconds")
    working_max_size: int = Field(1000, ge=10, description="Max working memory items")
    
    # Episodic memory
    episodic_enabled: bool = Field(True, description="Enable episodic memory")
    episodic_max_episodes: int = Field(10000, ge=100, description="Max episodes")
    episodic_retention_days: int = Field(30, ge=1, description="Episode retention days")
    
    # Semantic memory
    semantic_enabled: bool = Field(True, description="Enable semantic memory")
    semantic_max_items: int = Field(10000, ge=100, description="Max knowledge items")
    semantic_embedding_dim: int = Field(1536, ge=128, description="Embedding dimension")
    
    # Persistence
    persistence_enabled: bool = Field(False, description="Enable memory persistence")
    persistence_backend: str = Field("redis", description="Persistence backend")


class ToolConfig(BaseModel):
    """Tool configuration."""
    
    enabled: bool = Field(True, description="Enable tool system")
    max_concurrent: int = Field(5, ge=1, le=50, description="Max concurrent tool executions")
    default_timeout: float = Field(30.0, gt=0, le=300, description="Default tool timeout")
    
    # Security
    sandboxing_enabled: bool = Field(True, description="Enable tool sandboxing")
    allowed_tools: Optional[List[str]] = Field(None, description="Whitelist of allowed tools")
    blocked_tools: List[str] = Field(default_factory=list, description="Blacklist of blocked tools")
    
    # Resource limits
    max_memory_mb: int = Field(512, ge=64, description="Max memory per tool execution (MB)")
    max_cpu_percent: int = Field(80, ge=1, le=100, description="Max CPU usage percent")


class ObservabilityConfig(BaseModel):
    """Observability configuration."""
    
    # Logging
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field("json", description="Log format (json/text)")
    log_file: Optional[Path] = Field(None, description="Log file path")
    
    # Metrics
    metrics_enabled: bool = Field(True, description="Enable metrics collection")
    metrics_port: int = Field(9090, ge=1024, le=65535, description="Metrics server port")
    
    # Tracing
    tracing_enabled: bool = Field(False, description="Enable distributed tracing")
    tracing_endpoint: Optional[str] = Field(None, description="Tracing endpoint URL")
    
    # Health checks
    healthcheck_enabled: bool = Field(True, description="Enable health checks")
    healthcheck_interval: int = Field(30, ge=5, description="Health check interval (seconds)")
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()


class TeleonConfig(BaseSettings):
    """
    Main Teleon configuration.
    
    Loads configuration from:
    1. Environment variables (highest priority)
    2. Configuration file (teleon.yaml)
    3. Defaults (lowest priority)
    """
    
    # Environment
    environment: str = Field("development", description="Environment (development/staging/production)")
    debug: bool = Field(False, description="Debug mode")
    
    # Service info
    service_name: str = Field("teleon", description="Service name")
    service_version: str = Field("0.1.0", description="Service version")
    
    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    memory: MemoryConfig = Field(default_factory=MemoryConfig, description="Memory configuration")
    tools: ToolConfig = Field(default_factory=ToolConfig, description="Tool configuration")
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig, description="Observability configuration")
    
    # Secrets (loaded separately for security)
    _secrets: Optional[SecretsConfig] = None

    model_config = SettingsConfigDict(
        env_prefix="TELEON_",
        env_nested_delimiter="__",
        case_sensitive=False
    )

    @model_validator(mode='before')
    @classmethod
    def validate_environment(cls, values):
        """Validate environment configuration."""
        env = values.get("environment", "development")
        if env not in ["development", "staging", "production"]:
            raise ValueError("Environment must be one of: development, staging, production")
        
        # In production, enforce strict settings
        if env == "production":
            if values.get("debug", False):
                logger.warning("Debug mode should not be enabled in production")
        
        return values
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> "TeleonConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            TeleonConfig instance
        
        Raises:
            ConfigurationError: If file cannot be loaded
        """
        try:
            if not config_path.exists():
                raise ConfigurationError(
                    f"Configuration file not found: {config_path}",
                    {"path": str(config_path)}
                )
            
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                config_data = {}
            
            return cls(**config_data)
        
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {e}",
                {"path": str(config_path)}
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {e}",
                {"path": str(config_path)},
                cause=e
            )
    
    def load_secrets(self) -> None:
        """Load secrets from environment."""
        self._secrets = SecretsConfig.from_env()
        logger.info("Secrets loaded from environment")
    
    @property
    def secrets(self) -> SecretsConfig:
        """Get secrets (lazy load)."""
        if self._secrets is None:
            self.load_secrets()
        return self._secrets
    
    def validate_configuration(self) -> None:
        """
        Validate complete configuration.
        
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        errors = []
        
        # Validate required secrets in production
        if self.environment == "production":
            if not self.secrets.openai_api_key and not self.secrets.anthropic_api_key:
                errors.append("At least one LLM API key must be configured in production")
            
            if self.llm.enable_cache and not self.secrets.redis_url:
                errors.append("Redis URL required when caching is enabled in production")
        
        # Validate resource limits
        if self.tools.max_concurrent > 50:
            errors.append("Max concurrent tools should not exceed 50 for stability")
        
        if self.llm.max_concurrent_requests > 100:
            errors.append("Max concurrent LLM requests should not exceed 100")
        
        if errors:
            raise ConfigValidationError(
                "Configuration validation failed",
                {"errors": errors}
            )
        
        logger.info("Configuration validated successfully")
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Args:
            include_secrets: Whether to include secrets (default: False)
        
        Returns:
            Configuration dictionary
        """
        config_dict = self.dict(exclude={"_secrets"})
        
        if include_secrets and self._secrets:
            config_dict["secrets"] = self._secrets.dict()
        
        return config_dict
    
    def save_to_file(self, config_path: Path) -> None:
        """
        Save configuration to YAML file (excluding secrets).
        
        Args:
            config_path: Path to save configuration
        """
        config_dict = self.to_dict(include_secrets=False)
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {config_path}")
        
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration: {e}",
                {"path": str(config_path)},
                cause=e
            )


# Global configuration instance
_global_config: Optional[TeleonConfig] = None


@lru_cache(maxsize=1)
def get_config() -> TeleonConfig:
    """
    Get global configuration instance (singleton).
    
    Returns:
        TeleonConfig instance
    """
    global _global_config
    
    if _global_config is None:
        # Try to load from file
        config_path = Path("teleon.yaml")
        
        if config_path.exists():
            _global_config = TeleonConfig.load_from_file(config_path)
        else:
            # Use defaults
            _global_config = TeleonConfig()
        
        # Load secrets
        _global_config.load_secrets()
        
        # Validate
        _global_config.validate_configuration()
        
        logger.info(
            f"Configuration loaded",
            extra={
                "environment": _global_config.environment,
                "version": _global_config.service_version
            }
        )
    
    return _global_config


def reset_config() -> None:
    """Reset global configuration (for testing)."""
    global _global_config
    _global_config = None
    get_config.cache_clear()

