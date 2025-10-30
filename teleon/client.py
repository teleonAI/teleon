"""
Teleon Client - User authentication and agent registration
"""

import os
from typing import Optional, Dict, Any, Callable, List
import hashlib
from datetime import datetime
import httpx


class TeleonClient:
    """
    Teleon Client for authenticating users and registering agents.
    
    Usage:
        client = TeleonClient(api_key="your-teleon-api-key")
        
        @client.agent(name="my-agent")
        def my_agent(input: str) -> str:
            return "response"
    """
    
    # Global registry of all clients and their agents
    _instances: Dict[str, 'TeleonClient'] = {}
    _all_agents: Dict[str, Dict[str, Any]] = {}
    
    def __init__(
        self,
        api_key: str,
        environment: str = "production",
        base_url: Optional[str] = None,
        verify_key: bool = True
    ):
        """
        Initialize Teleon client.
        
        Args:
            api_key: Your Teleon API key from the platform
            environment: Environment (production, staging, dev)
            base_url: Custom API base URL (default: uses environment)
            verify_key: Whether to verify API key with backend (default: True)
        """
        self.api_key = api_key
        self.environment = environment
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.scopes: List[str] = []  # API key scopes/permissions
        
        # Validate API key format
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Valid Teleon API key is required")
        
        # Validate API key format pattern
        if not api_key.startswith(('tlk_live_', 'tlk_test_', 'teleon_')):
            raise ValueError(
                f"Invalid API key format. Expected format: tlk_live_xxx or tlk_test_xxx\n"
                f"Get your API key from: https://dashboard.teleon.ai"
            )
        
        # Set base URL based on environment
        if base_url:
            self.base_url = base_url
        elif environment == "production":
            self.base_url = "https://api.teleon.ai"
        elif environment == "staging":
            self.base_url = "https://api.staging.teleon.ai"
        else:  # dev
            self.base_url = "http://localhost:8000"
        
        # Generate user ID from API key
        self.user_id = self._generate_user_id(api_key)
        
        # Initialize HTTP client for remote operations
        self._http_client = None
        
        # Verify API key with backend if requested
        if verify_key and environment != "dev":
            self._verify_api_key_sync()
        
        # Register this client instance
        TeleonClient._instances[self.user_id] = self
        
        # Only print if not in quiet mode (used during agent discovery)
        if not os.getenv('TELEON_QUIET'):
            print(f"✓ Teleon Client initialized")
            print(f"  User ID: {self.user_id}")
            print(f"  Environment: {environment}")
            print(f"  API URL: {self.base_url}")
    
    def _generate_user_id(self, api_key: str) -> str:
        """Generate a user ID from API key."""
        return hashlib.sha256(api_key.encode()).hexdigest()[:12]
    
    def has_scope(self, *required_scopes: str) -> bool:
        """
        Check if API key has any of the required scopes.
        
        Args:
            *required_scopes: One or more required scopes (ANY match returns True)
        
        Returns:
            True if key has at least one of the required scopes
        
        Example:
            if not client.has_scope('agents:deploy'):
                raise ValueError("API key needs 'agents:deploy' permission")
        """
        if not self.scopes:
            # If scopes weren't loaded, assume dev mode or old key
            return True
        
        return any(scope in self.scopes for scope in required_scopes)
    
    def require_scope(self, *required_scopes: str):
        """
        Raise an error if API key doesn't have required scopes.
        
        Args:
            *required_scopes: One or more required scopes (ANY match is OK)
        
        Raises:
            ValueError: If key doesn't have any of the required scopes
        
        Example:
            client.require_scope('agents:deploy')  # Will raise if missing
        """
        if not self.has_scope(*required_scopes):
            raise ValueError(
                f"Insufficient API key permissions. Required scopes: {', '.join(required_scopes)}. "
                f"Your key has: {', '.join(self.scopes) if self.scopes else 'none'}. "
                f"Please create a new API key with the required permissions at https://dashboard.teleon.ai"
            )
    
    def _verify_api_key_sync(self):
        """
        Verify API key with the backend (synchronous version for __init__).
        
        This makes a simple API call to validate the key is real and active.
        """
        import time
        
        try:
            # Use synchronous httpx client for initialization
            with httpx.Client(timeout=10.0) as client:
                # Try to validate the API key using the validate endpoint
                response = client.get(
                    f"{self.base_url}/api/v1/api-keys/validate",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                # Check if authentication succeeded
                if response.status_code == 401:
                    try:
                        error_detail = response.json().get("detail", "Invalid API key")
                    except:
                        error_detail = "Invalid API key"
                    raise ValueError(
                        f"API key verification failed: {error_detail}\n"
                        f"Please check your API key and try again.\n"
                        f"Get a valid API key from: https://dashboard.teleon.ai"
                    )
                elif response.status_code == 403:
                    raise ValueError(
                        f"API key is valid but your account is inactive.\n"
                        f"Please contact support or check your account status."
                    )
                elif response.status_code == 429:
                    raise ValueError(
                        f"Rate limit exceeded. Please wait a moment and try again."
                    )
                elif response.status_code >= 500:
                    # Server error - allow to proceed but warn
                    print(f"  ⚠️  Warning: Unable to verify API key (server error)")
                    return
                elif response.status_code >= 400:
                    raise ValueError(
                        f"API key verification failed with status {response.status_code}\n"
                        f"Response: {response.text}"
                    )
                
                # Success! Get API key info including scopes
                if response.status_code == 200:
                    try:
                        key_info = response.json()
                        self.scopes = key_info.get("key", {}).get("scopes", [])
                        if not os.getenv('TELEON_QUIET'):
                            print(f"  ✓ API key verified successfully")
                            if self.scopes:
                                print(f"  ℹ️  Scopes: {', '.join(self.scopes)}")
                            else:
                                print(f"  ⚠️  Warning: No scopes assigned to this API key")
                    except Exception:
                        # Couldn't parse scopes but key is valid
                        if not os.getenv('TELEON_QUIET'):
                            print(f"  ✓ API key verified successfully")
                else:
                    if not os.getenv('TELEON_QUIET'):
                        print(f"  ✓ API key verified")
                
        except httpx.ConnectError:
            # Can't connect to server - allow in dev mode
            if not os.getenv('TELEON_QUIET'):
                print(f"  ⚠️  Warning: Could not connect to {self.base_url}")
                print(f"  Skipping API key verification (server may not be running)")
        except httpx.TimeoutException:
            if not os.getenv('TELEON_QUIET'):
                print(f"  ⚠️  Warning: Connection timeout")
                print(f"  Skipping API key verification")
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Other errors - warn but don't block
            if not os.getenv('TELEON_QUIET'):
                print(f"  ⚠️  Warning: Could not verify API key: {str(e)}")
    
    def agent(
        self,
        name: str,
        description: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 500,
        helix: Optional[Dict[str, Any]] = None,
        cortex: Optional[Dict[str, Any]] = None,
        nexusnet: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Decorator to register an agent with this client.
        
        Args:
            name: Agent name
            description: Agent description
            model: LLM model to use
            temperature: Temperature setting
            max_tokens: Max tokens
            helix: Helix runtime configuration (auto-scaling, health checks)
            cortex: Cortex memory configuration (learning, memory types)
            nexusnet: NexusNet collaboration configuration (multi-agent)
            **kwargs: Additional configuration
        
        Returns:
            Decorated function
        
        Example:
            ```python
            @client.agent(
                name="my-agent",
                helix={'min': 2, 'max': 10, 'target_cpu': 70},
                cortex={'learning': True, 'memory_types': ['episodic', 'semantic']},
                nexusnet={'capabilities': ['research'], 'collaborate': True}
            )
            async def my_agent(input: str):
                return process(input)
            ```
        """
        def decorator(func: Callable):
            # Generate unique agent ID
            agent_id = self._generate_agent_id(name)
            
            # Extract function signature for OpenAPI
            import inspect
            sig = inspect.signature(func)
            params = {}
            for param_name, param in sig.parameters.items():
                params[param_name] = {
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "string",
                    "required": param.default == inspect.Parameter.empty
                }
            
            # Register agent
            agent_info = {
                "agent_id": agent_id,
                "name": name,
                "description": description or func.__doc__ or "No description provided",
                "function": func,
                "user_id": self.user_id,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "parameters": params,
                "created_at": datetime.utcnow().isoformat(),
                "helix": helix,
                "cortex": cortex,
                "nexusnet": nexusnet,
                "config": kwargs
            }
            
            self.agents[agent_id] = agent_info
            TeleonClient._all_agents[agent_id] = agent_info
            
            if not os.getenv('TELEON_QUIET'):
                print(f"✓ Agent registered: {name}")
                print(f"  Agent ID: {agent_id}")
                print(f"  URL: /{agent_id}/")
            
            return func
        
        return decorator
    
    def _generate_agent_id(self, name: str) -> str:
        """Generate a unique agent ID."""
        # Combine user ID and agent name for uniqueness
        unique_str = f"{self.user_id}:{name}:{datetime.utcnow().isoformat()}"
        hash_id = hashlib.sha256(unique_str.encode()).hexdigest()[:16]
        return f"agent_{hash_id}"
    
    @classmethod
    def get_all_agents(cls) -> Dict[str, Dict[str, Any]]:
        """Get all registered agents across all clients."""
        return cls._all_agents
    
    @classmethod
    def get_agent(cls, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent by ID."""
        return cls._all_agents.get(agent_id)
    
    @classmethod
    def get_client(cls, user_id: str) -> Optional['TeleonClient']:
        """Get a client by user ID."""
        return cls._instances.get(user_id)
    
    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for remote operations."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=httpx.Timeout(60.0)
            )
        return self._http_client
    
    async def get_agent(self, agent_name: str) -> 'RemoteAgent':
        """
        Get a reference to a deployed agent.
        
        This returns a RemoteAgent proxy that can be used to interact with
        the deployed agent from your application.
        
        Args:
            agent_name: Name of the deployed agent
        
        Returns:
            RemoteAgent proxy for interacting with the agent
        
        Example:
            client = TeleonClient(api_key="...")
            agent = await client.get_agent("customer-support")
            result = await agent.execute(input_data={...})
        """
        from teleon.remote_agent import RemoteAgent
        
        # Fetch agent info from API
        client = self._get_http_client()
        
        try:
            # Search for agent by name
            response = await client.get(
                f"{self.base_url}/agents",
                params={"name": agent_name, "user_id": self.user_id}
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("agents"):
                raise ValueError(f"Agent '{agent_name}' not found. Make sure it's deployed.")
            
            agent_data = data["agents"][0]
            agent_id = agent_data["agent_id"]
            
            # Return RemoteAgent proxy
            return RemoteAgent(
                agent_name=agent_name,
                agent_id=agent_id,
                api_key=self.api_key,
                base_url=self.base_url
            )
            
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Failed to fetch agent: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise ValueError(f"Failed to fetch agent: {str(e)}")
    
    async def list_deployed_agents(self) -> List[Dict[str, Any]]:
        """
        List all deployed agents for this user.
        
        Returns:
            List of deployed agents with their info
        """
        client = self._get_http_client()
        
        try:
            response = await client.get(
                f"{self.base_url}/agents",
                params={"user_id": self.user_id}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("agents", [])
            
        except Exception as e:
            raise ValueError(f"Failed to list agents: {str(e)}")
    
    async def close(self):
        """Close HTTP client connections."""
        if self._http_client:
            await self._http_client.aclose()
    
    async def initialize_agent_runtime(
        self,
        agent_id: str,
        enable_helix: bool = True,
        enable_cortex: bool = True,
        enable_nexusnet: bool = True
    ) -> Dict[str, Any]:
        """
        Initialize runtime features for an agent.
        
        This sets up Helix, Cortex, and NexusNet for the agent based on
        its configuration.
        
        Args:
            agent_id: Agent ID to initialize
            enable_helix: Enable Helix runtime
            enable_cortex: Enable Cortex memory
            enable_nexusnet: Enable NexusNet collaboration
        
        Returns:
            Dictionary with initialized components
        
        Example:
            ```python
            components = await client.initialize_agent_runtime("agent-123")
            cortex = components['cortex']
            runtime = components['helix_runtime']
            ```
        """
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            raise ValueError(f"Agent {agent_id} not found")
        
        components = {}
        
        # Initialize Helix if configured
        if enable_helix and agent_info.get('helix'):
            from teleon.helix.integration import register_agent_with_helix, parse_helix_config
            
            helix_config = parse_helix_config(agent_info['helix'])
            
            # Register with Helix (will create wrapper)
            wrapper = await register_agent_with_helix(
                agent_id=agent_id,
                agent_func=agent_info['function'],
                helix_config=helix_config
            )
            
            components['helix_wrapper'] = wrapper
            components['helix_config'] = helix_config
        
        # Initialize Cortex if configured
        if enable_cortex and agent_info.get('cortex'):
            from teleon.cortex import create_cortex, CortexConfig
            
            cortex_config_dict = agent_info['cortex']
            
            # Create Cortex instance
            cortex = await create_cortex(
                agent_id=agent_id,
                session_id=self.user_id,
                storage_backend=cortex_config_dict.get('storage', 'memory'),
                config=CortexConfig(**cortex_config_dict) if isinstance(cortex_config_dict, dict) else None
            )
            
            components['cortex'] = cortex
        
        # Initialize NexusNet if configured
        if enable_nexusnet and agent_info.get('nexusnet'):
            from teleon.nexusnet import get_registry
            
            nexusnet_config = agent_info['nexusnet']
            registry = get_registry()
            
            # Register capabilities
            if 'capabilities' in nexusnet_config:
                # Register agent with NexusNet
                await registry.register(
                    agent_id=agent_id,
                    name=agent_info['name'],
                    capabilities=nexusnet_config['capabilities'],  # List of strings
                    description=agent_info['description']
                )
            
            components['nexusnet_registry'] = registry
            components['nexusnet_config'] = nexusnet_config
        
        return components


# Convenience function for quick setup
def init_teleon(api_key: str, environment: str = "dev") -> TeleonClient:
    """
    Initialize Teleon client.
    
    Args:
        api_key: Your Teleon API key
        environment: Environment (dev, staging, production)
    
    Returns:
        TeleonClient instance
    """
    return TeleonClient(api_key=api_key, environment=environment)

