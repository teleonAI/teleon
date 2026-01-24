"""
Context-Aware Request Routing.

This module provides intelligent routing of requests based on context window
requirements, routing to appropriate model variants and instances.
"""

from typing import Dict, Optional, List, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import random

from teleon.helix.llm_metrics import TokenCounter
from teleon.core import (
    StructuredLogger,
    LogLevel,
)


class ModelVariant(str, Enum):
    """Model variants based on context window size."""
    SMALL = "small"      # Up to 4K tokens
    MEDIUM = "medium"    # 4K-32K tokens
    LARGE = "large"      # 32K-128K tokens
    XLARGE = "xlarge"    # 128K+ tokens


class ContextWindowConfig(BaseModel):
    """Configuration for a context window tier."""
    
    variant: ModelVariant = Field(..., description="Variant identifier")
    max_context_tokens: int = Field(..., gt=0, description="Max context size")
    models: List[str] = Field(..., description="Supported models")
    cost_multiplier: float = Field(1.0, gt=0, description="Cost relative to small")
    latency_multiplier: float = Field(1.0, gt=0, description="Latency relative to small")


# Default context window configurations
DEFAULT_CONTEXT_WINDOWS = [
    ContextWindowConfig(
        variant=ModelVariant.SMALL,
        max_context_tokens=4096,
        models=["gpt-3.5-turbo", "claude-3-haiku"],
        cost_multiplier=1.0,
        latency_multiplier=1.0
    ),
    ContextWindowConfig(
        variant=ModelVariant.MEDIUM,
        max_context_tokens=32768,
        models=["gpt-4", "gpt-4-turbo", "claude-3-sonnet"],
        cost_multiplier=3.0,
        latency_multiplier=1.5
    ),
    ContextWindowConfig(
        variant=ModelVariant.LARGE,
        max_context_tokens=128000,
        models=["gpt-4-turbo-preview", "claude-3-opus"],
        cost_multiplier=5.0,
        latency_multiplier=2.0
    ),
    ContextWindowConfig(
        variant=ModelVariant.XLARGE,
        max_context_tokens=200000,
        models=["claude-3-opus", "claude-3-sonnet"],
        cost_multiplier=7.0,
        latency_multiplier=2.5
    ),
]


class RoutingDecision(BaseModel):
    """Routing decision result."""
    
    instance_id: Optional[str] = Field(None, description="Selected instance")
    model: str = Field(..., description="Selected model")
    variant: ModelVariant = Field(..., description="Model variant")
    estimated_cost: float = Field(0.0, description="Estimated cost")
    estimated_latency_ms: float = Field(0.0, description="Estimated latency")
    reason: str = Field("", description="Routing reason")
    fallback: bool = Field(False, description="Whether this is a fallback")


class InstanceInfo(BaseModel):
    """Information about an agent instance."""
    
    instance_id: str = Field(..., description="Instance identifier")
    agent_id: str = Field(..., description="Agent identifier")
    model: str = Field(..., description="Model running on instance")
    variant: ModelVariant = Field(..., description="Model variant")
    max_context_tokens: int = Field(..., description="Max context size")
    
    # Load information
    current_load: float = Field(0.0, ge=0, le=1, description="Current load (0-1)")
    queue_depth: int = Field(0, ge=0, description="Queue depth")
    
    # Performance
    avg_latency_ms: float = Field(0.0, description="Average latency")
    
    # Availability
    available: bool = Field(True, description="Instance available")
    healthy: bool = Field(True, description="Instance healthy")


class ContextWindowManager:
    """
    Manage instances by context window capacity.
    
    Routes requests to appropriate instances based on context size
    requirements and instance capabilities.
    """
    
    def __init__(
        self,
        context_windows: Optional[List[ContextWindowConfig]] = None
    ):
        """
        Initialize context window manager.
        
        Args:
            context_windows: Context window configurations
        """
        self.context_windows = context_windows or DEFAULT_CONTEXT_WINDOWS
        self.context_windows.sort(key=lambda x: x.max_context_tokens)
        
        # Instance registry by variant
        self.instances_by_variant: Dict[ModelVariant, List[InstanceInfo]] = {
            variant: [] for variant in ModelVariant
        }
        
        # Token counter
        self.token_counter = TokenCounter()
        
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger("context_window_manager", LogLevel.INFO)
    
    async def register_instance(
        self,
        instance_id: str,
        agent_id: str,
        model: str,
        max_context_tokens: int
    ):
        """
        Register an instance with its context capacity.
        
        Args:
            instance_id: Instance identifier
            agent_id: Agent identifier
            model: Model name
            max_context_tokens: Max context size
        """
        # Determine variant
        variant = self._determine_variant(max_context_tokens)
        
        instance_info = InstanceInfo(
            instance_id=instance_id,
            agent_id=agent_id,
            model=model,
            variant=variant,
            max_context_tokens=max_context_tokens
        )
        
        async with self.lock:
            self.instances_by_variant[variant].append(instance_info)
            
            self.logger.info(
                f"Registered instance {instance_id} as {variant.value} "
                f"(max {max_context_tokens} tokens)"
            )
    
    async def unregister_instance(self, instance_id: str):
        """Unregister an instance."""
        async with self.lock:
            for variant in ModelVariant:
                self.instances_by_variant[variant] = [
                    inst for inst in self.instances_by_variant[variant]
                    if inst.instance_id != instance_id
                ]
            
            self.logger.info(f"Unregistered instance {instance_id}")
    
    async def update_instance_load(
        self,
        instance_id: str,
        load: float,
        queue_depth: int,
        avg_latency_ms: float
    ):
        """Update instance load metrics."""
        async with self.lock:
            for variant in ModelVariant:
                for instance in self.instances_by_variant[variant]:
                    if instance.instance_id == instance_id:
                        instance.current_load = load
                        instance.queue_depth = queue_depth
                        instance.avg_latency_ms = avg_latency_ms
                        return
    
    def _determine_variant(self, context_tokens: int) -> ModelVariant:
        """Determine variant for context size."""
        for window in self.context_windows:
            if context_tokens <= window.max_context_tokens:
                return window.variant
        
        # If larger than all windows, use largest
        return self.context_windows[-1].variant
    
    def _get_window_config(self, variant: ModelVariant) -> Optional[ContextWindowConfig]:
        """Get window configuration for variant."""
        for window in self.context_windows:
            if window.variant == variant:
                return window
        return None
    
    async def get_best_instance(
        self,
        required_context_tokens: int,
        prefer_model: Optional[str] = None,
        prefer_cost: bool = True
    ) -> Optional[InstanceInfo]:
        """
        Get best instance for required context size.
        
        Args:
            required_context_tokens: Required context size
            prefer_model: Preferred model name
            prefer_cost: Prefer lower cost
        
        Returns:
            Best instance or None
        """
        async with self.lock:
            # Determine required variant
            required_variant = self._determine_variant(required_context_tokens)
            
            # Get candidates (this variant and larger)
            candidates = []
            for window in self.context_windows:
                if window.variant.value >= required_variant.value:
                    candidates.extend(self.instances_by_variant[window.variant])
            
            if not candidates:
                return None
            
            # Filter to available and healthy
            candidates = [
                inst for inst in candidates
                if inst.available and inst.healthy
            ]
            
            if not candidates:
                return None
            
            # Filter by model preference if specified
            if prefer_model:
                model_matches = [
                    inst for inst in candidates
                    if prefer_model.lower() in inst.model.lower()
                ]
                if model_matches:
                    candidates = model_matches
            
            # Score candidates
            scored_candidates = []
            for inst in candidates:
                score = self._score_instance(inst, required_variant, prefer_cost)
                scored_candidates.append((inst, score))
            
            # Sort by score (higher is better)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            return scored_candidates[0][0]
    
    def _score_instance(
        self,
        instance: InstanceInfo,
        required_variant: ModelVariant,
        prefer_cost: bool
    ) -> float:
        """
        Score an instance for routing.
        
        Higher score = better candidate
        """
        score = 100.0
        
        # Prefer exact variant match (lower cost)
        if instance.variant == required_variant:
            score += 20.0
        else:
            # Penalize over-provisioned instances
            variant_diff = list(ModelVariant).index(instance.variant) - \
                          list(ModelVariant).index(required_variant)
            score -= variant_diff * 10.0
        
        # Prefer lower load
        score -= instance.current_load * 30.0
        
        # Prefer lower queue depth
        score -= min(instance.queue_depth, 10) * 2.0
        
        # Consider latency
        if instance.avg_latency_ms > 0:
            # Normalize to ~0-10 penalty
            latency_penalty = min(instance.avg_latency_ms / 1000.0, 10.0)
            score -= latency_penalty
        
        # Cost consideration
        if prefer_cost:
            window_config = self._get_window_config(instance.variant)
            if window_config:
                score -= (window_config.cost_multiplier - 1.0) * 5.0
        
        return score
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about instance distribution."""
        async with self.lock:
            stats = {
                "total_instances": sum(
                    len(instances) for instances in self.instances_by_variant.values()
                ),
                "by_variant": {}
            }
            
            for variant in ModelVariant:
                instances = self.instances_by_variant[variant]
                if instances:
                    avg_load = sum(i.current_load for i in instances) / len(instances)
                    total_queue = sum(i.queue_depth for i in instances)
                    available_count = sum(1 for i in instances if i.available)
                    
                    stats["by_variant"][variant.value] = {
                        "count": len(instances),
                        "available": available_count,
                        "avg_load": avg_load,
                        "total_queue": total_queue
                    }
                else:
                    stats["by_variant"][variant.value] = {
                        "count": 0,
                        "available": 0,
                        "avg_load": 0.0,
                        "total_queue": 0
                    }
            
            return stats


class ModelVariantRouter:
    """
    Route requests to appropriate model variants.
    
    Combines context size analysis with cost/performance preferences
    to select optimal model variant.
    """
    
    def __init__(
        self,
        context_manager: ContextWindowManager
    ):
        """
        Initialize model variant router.
        
        Args:
            context_manager: Context window manager
        """
        self.context_manager = context_manager
        self.token_counter = TokenCounter()
        self.logger = StructuredLogger("variant_router", LogLevel.INFO)
    
    async def route_request(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        prefer_cost: bool = True,
        prefer_speed: bool = False
    ) -> RoutingDecision:
        """
        Route request to best instance.
        
        Args:
            messages: Conversation messages
            model: Preferred model
            max_tokens: Max response tokens
            prefer_cost: Prefer lower cost
            prefer_speed: Prefer lower latency
        
        Returns:
            Routing decision
        """
        # Count input tokens
        input_tokens = self.token_counter.count_messages_tokens(
            messages,
            model or "gpt-4"
        )
        
        # Estimate output tokens
        output_tokens = max_tokens or 500
        
        # Total context requirement
        required_tokens = input_tokens + output_tokens
        
        self.logger.debug(
            f"Routing request: {input_tokens} input + {output_tokens} output "
            f"= {required_tokens} total tokens"
        )
        
        # Get best instance
        instance = await self.context_manager.get_best_instance(
            required_context_tokens=required_tokens,
            prefer_model=model,
            prefer_cost=prefer_cost
        )
        
        if not instance:
            # No instance available - return fallback decision
            self.logger.warning(
                "No instances available for routing",
                required_tokens=required_tokens,
                model=model
            )
            return RoutingDecision(
                instance_id=None,
                model=model or "gpt-4",
                variant=ModelVariant.MEDIUM,
                reason="No instances available - fallback",
                fallback=True
            )
        
        # Validate instance is still available and healthy
        async with self.context_manager.lock:
            # Re-check instance availability
            instance_found = False
            for variant in ModelVariant:
                for inst in self.context_manager.instances_by_variant[variant]:
                    if inst.instance_id == instance.instance_id:
                        if not inst.available or not inst.healthy:
                            self.logger.warning(
                                "Selected instance no longer available",
                                instance_id=instance.instance_id
                            )
                            # Try to get another instance
                            instance = await self.context_manager.get_best_instance(
                                required_context_tokens=required_tokens,
                                prefer_model=model,
                                prefer_cost=prefer_cost
                            )
                            if not instance:
                                return RoutingDecision(
                                    instance_id=None,
                                    model=model or "gpt-4",
                                    variant=ModelVariant.MEDIUM,
                                    reason="No healthy instances available - fallback",
                                    fallback=True
                                )
                        instance_found = True
                        break
                if instance_found:
                    break
            
            if not instance_found:
                self.logger.error(
                    "Selected instance not found in registry",
                    instance_id=instance.instance_id
                )
                return RoutingDecision(
                    instance_id=None,
                    model=model or "gpt-4",
                    variant=ModelVariant.MEDIUM,
                    reason="Instance validation failed - fallback",
                    fallback=True
                )
        
        # Estimate cost and latency
        window_config = self.context_manager._get_window_config(instance.variant)
        
        if window_config:
            base_cost = 0.001  # Base cost estimate
            estimated_cost = base_cost * window_config.cost_multiplier
            
            base_latency = 1000.0  # Base latency estimate (ms)
            estimated_latency = base_latency * window_config.latency_multiplier
        else:
            estimated_cost = 0.0
            estimated_latency = 0.0
        
        reason = (
            f"Routed to {instance.variant.value} instance "
            f"({required_tokens} tokens, load={instance.current_load:.2f})"
        )
        
        return RoutingDecision(
            instance_id=instance.instance_id,
            model=instance.model,
            variant=instance.variant,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            reason=reason,
            fallback=False
        )


class RequestRouter:
    """
    High-level request router.
    
    Combines context-aware routing with load balancing and
    failover capabilities.
    """
    
    def __init__(
        self,
        context_manager: Optional[ContextWindowManager] = None
    ):
        """
        Initialize request router.
        
        Args:
            context_manager: Context window manager
        """
        self.context_manager = context_manager or ContextWindowManager()
        self.variant_router = ModelVariantRouter(self.context_manager)
        
        # Routing statistics
        self.total_routes = 0
        self.routes_by_variant: Dict[ModelVariant, int] = {
            v: 0 for v in ModelVariant
        }
        self.fallback_count = 0
        
        self.logger = StructuredLogger("request_router", LogLevel.INFO)
    
    async def route(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> RoutingDecision:
        """
        Route a request.
        
        Args:
            messages: Conversation messages
            model: Preferred model
            max_tokens: Max response tokens
            **kwargs: Additional routing preferences
        
        Returns:
            Routing decision
        """
        decision = await self.variant_router.route_request(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            prefer_cost=kwargs.get("prefer_cost", True),
            prefer_speed=kwargs.get("prefer_speed", False)
        )
        
        # Update statistics
        self.total_routes += 1
        self.routes_by_variant[decision.variant] += 1
        
        if decision.fallback:
            self.fallback_count += 1
        
        self.logger.info(
            decision.reason,
            instance_id=decision.instance_id,
            model=decision.model,
            variant=decision.variant.value
        )
        
        return decision
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "total_routes": self.total_routes,
            "fallback_count": self.fallback_count,
            "fallback_rate": self.fallback_count / max(self.total_routes, 1),
            "routes_by_variant": {
                v.value: count for v, count in self.routes_by_variant.items()
            }
        }


# Global router
_global_router: Optional[RequestRouter] = None


def get_request_router() -> RequestRouter:
    """Get global request router."""
    global _global_router
    if _global_router is None:
        _global_router = RequestRouter()
    return _global_router

