"""
Helix Runtime - Production-grade agent runtime system.

This package provides local development runtime with:
- Agent process management
- Resource configuration and limits
- Health and readiness checks
- Hot reload for development
- Container support
- Scaling management
- LLM-specific features (metrics, token tracking, caching)
"""

from teleon.helix.runtime import AgentRuntime, RuntimeConfig, ResourceConfig, get_runtime
from teleon.helix.process import ProcessManager, ProcessInfo, ProcessStatus
from teleon.helix.health import HealthChecker, HealthCheck, HealthStatus, CheckType
from teleon.helix.scaling import Scaler, ScalingPolicy, ScalingMetrics
from teleon.helix.watcher import FileWatcher, WatcherConfig

# LLM-specific imports
from teleon.helix.llm_metrics import (
    LLMMetrics,
    LLMResourceTracker,
    TokenCounter,
    TokenThroughputMonitor,
    get_throughput_monitor,
)
from teleon.helix.llm_scaling import (
    TokenAwareScalingPolicy,
    CostAwareScalingPolicy,
    LLMScaler,
    get_llm_scaler,
    create_llm_scaler,
)
from teleon.helix.cost_tracker import (
    TokenTracker,
    TokenBreakdown,
    TokenPeriod,
    TokenAlert,
    TokenBudgetManager,
    get_token_tracker,
    get_token_budget_manager,
    get_cost_tracker,  # Backward compatibility alias
)
from teleon.helix.context_router import (
    ContextWindowManager,
    ModelVariantRouter,
    RequestRouter,
    ModelVariant,
    RoutingDecision,
    get_request_router,
)
from teleon.helix.batch_processor import (
    BatchProcessor,
    RequestBatcher,
    BatchOptimizer,
    BatchConfig,
    get_batch_processor,
)
from teleon.helix.cache_manager import (
    ResponseCache,
    CacheStrategy,
    CacheEvictionPolicy,
    CacheEntry,
    get_response_cache,
    create_cache,
)
from teleon.helix.agent_reporter import (
    AgentMetricsReporter,
    get_agent_reporter,
    init_agent_reporter,
    shutdown_agent_reporter,
)
from teleon.helix.health_endpoints import (
    HealthEndpointManager,
    HealthStatus as EndpointHealthStatus,
    get_health_manager,
    setup_health_endpoints,
)

__all__ = [
    # Runtime
    "AgentRuntime",
    "RuntimeConfig",
    "ResourceConfig",
    "get_runtime",
    
    # Process Management
    "ProcessManager",
    "ProcessInfo",
    "ProcessStatus",
    
    # Health Checks
    "HealthChecker",
    "HealthCheck",
    "HealthStatus",
    "CheckType",
    
    # Scaling
    "Scaler",
    "ScalingPolicy",
    "ScalingMetrics",
    
    # Hot Reload
    "FileWatcher",
    "WatcherConfig",
    
    # LLM Metrics
    "LLMMetrics",
    "LLMResourceTracker",
    "TokenCounter",
    "TokenThroughputMonitor",
    "get_throughput_monitor",
    
    # LLM Scaling
    "TokenAwareScalingPolicy",
    "CostAwareScalingPolicy",
    "LLMScaler",
    "get_llm_scaler",
    "create_llm_scaler",
    
    # Token Tracking
    "TokenTracker",
    "TokenBreakdown",
    "TokenPeriod",
    "TokenAlert",
    "TokenBudgetManager",
    "get_token_tracker",
    "get_token_budget_manager",
    "get_cost_tracker",  # Backward compatibility
    
    # Context Routing
    "ContextWindowManager",
    "ModelVariantRouter",
    "RequestRouter",
    "ModelVariant",
    "RoutingDecision",
    "get_request_router",
    
    # Batch Processing
    "BatchProcessor",
    "RequestBatcher",
    "BatchOptimizer",
    "BatchConfig",
    "get_batch_processor",
    
    # Caching
    "ResponseCache",
    "CacheStrategy",
    "CacheEvictionPolicy",
    "CacheEntry",
    "get_response_cache",
    "create_cache",
    
    # Agent Metrics Reporter
    "AgentMetricsReporter",
    "get_agent_reporter",
    "init_agent_reporter",
    "shutdown_agent_reporter",
    
    # Health Endpoints
    "HealthEndpointManager",
    "EndpointHealthStatus",
    "get_health_manager",
    "setup_health_endpoints",
]

