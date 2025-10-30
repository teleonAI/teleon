"""Configuration loader for Teleon agents."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from teleon.config.agent_config import AgentConfig


class ConfigLoader:
    """
    Load configuration from multiple sources with precedence:
    1. Explicit parameters (highest priority)
    2. Environment variables
    3. teleon.yaml file
    4. Default values (lowest priority)
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_file: Path to configuration file (defaults to teleon.yaml)
        """
        self.config_file = config_file or Path("teleon.yaml")
        self._config_data: Optional[Dict[str, Any]] = None
        
        # Load environment variables from .env file
        load_dotenv()
    
    def load(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration with precedence rules.
        
        Args:
            agent_name: Optional agent name to load specific config
        
        Returns:
            Merged configuration dictionary
        """
        config = {}
        
        # 1. Load from file
        file_config = self._load_from_file()
        if file_config:
            config.update(file_config)
            
            # If agent name provided, get agent-specific config
            if agent_name and 'agents' in file_config:
                agent_config = file_config['agents'].get(agent_name, {})
                config.update(agent_config)
        
        # 2. Override with environment variables
        env_config = self._load_from_env()
        config.update(env_config)
        
        return config
    
    def _load_from_file(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                data = yaml.safe_load(f)
                return data or {}
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_file}: {e}")
            return {}
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Database URLs
        if db_url := os.getenv('DATABASE_URL'):
            config['database_url'] = db_url
        
        if redis_url := os.getenv('REDIS_URL'):
            config['redis_url'] = redis_url
        
        if chromadb_url := os.getenv('CHROMADB_URL'):
            config['chromadb_url'] = chromadb_url
        
        # LLM API Keys
        llm_keys = {}
        if openai_key := os.getenv('OPENAI_API_KEY'):
            llm_keys['openai'] = openai_key
        
        if anthropic_key := os.getenv('ANTHROPIC_API_KEY'):
            llm_keys['anthropic'] = anthropic_key
        
        if google_key := os.getenv('GOOGLE_API_KEY'):
            llm_keys['google'] = google_key
        
        if llm_keys:
            config['llm_api_keys'] = llm_keys
        
        # Feature flags
        if env_memory := os.getenv('ENABLE_MEMORY'):
            config['enable_memory'] = env_memory.lower() == 'true'
        
        if env_learning := os.getenv('ENABLE_LEARNING'):
            config['enable_learning'] = env_learning.lower() == 'true'
        
        if env_collab := os.getenv('ENABLE_COLLABORATION'):
            config['enable_collaboration'] = env_collab.lower() == 'true'
        
        # Environment
        if environment := os.getenv('ENVIRONMENT'):
            config['environment'] = environment
        
        if debug := os.getenv('DEBUG'):
            config['debug'] = debug.lower() == 'true'
        
        return config
    
    def load_agent_config(
        self,
        agent_name: str,
        **overrides: Any
    ) -> AgentConfig:
        """
        Load and create AgentConfig for a specific agent.
        
        Args:
            agent_name: Name of the agent
            **overrides: Configuration overrides
        
        Returns:
            AgentConfig instance
        """
        # Load base configuration
        config = self.load(agent_name)
        
        # Apply overrides
        config.update(overrides)
        
        # Create AgentConfig
        return AgentConfig(
            name=agent_name,
            memory=config.get('memory', False),
            scale=config.get('scale', {'min': 1, 'max': 10}),
            llm=config.get('llm', {}),
            tools=config.get('tools', []),
            collaborate=config.get('collaborate', False),
            timeout=config.get('timeout'),
        )


def load_config(
    config_file: Optional[Path] = None,
    agent_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        config_file: Path to configuration file
        agent_name: Optional agent name
    
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_file)
    return loader.load(agent_name)

