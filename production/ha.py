"""
High Availability - Production-grade HA features.

Features:
- Multi-region deployment
- Load balancing
- Circuit breaker
- Automatic failover
- Health-based routing
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import random

from teleon.core import StructuredLogger, LogLevel, get_metrics


class LoadBalancingAlgorithm(str, Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    RANDOM = "random"
    LEAST_RESPONSE_TIME = "least_response_time"


class FailoverStrategy(str, Enum):
    """Failover strategies."""
    ACTIVE_ACTIVE = "active_active"
    ACTIVE_PASSIVE = "active_passive"
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class HAConfig(BaseModel):
    """High availability configuration."""
    
    # Multi-region
    regions: List[str] = Field(default_factory=list, description="Deployment regions")
    strategy: FailoverStrategy = Field(FailoverStrategy.AUTOMATIC, description="Failover strategy")
    
    # Load balancing
    load_balancing_algorithm: LoadBalancingAlgorithm = Field(
        LoadBalancingAlgorithm.ROUND_ROBIN,
        description="Load balancing algorithm"
    )
    
    # Health checks
    health_check_interval: int = Field(30, ge=5, description="Health check interval (seconds)")
    unhealthy_threshold: int = Field(3, ge=1, description="Failures before unhealthy")
    healthy_threshold: int = Field(2, ge=1, description="Successes before healthy")
    
    # Circuit breaker
    circuit_breaker_enabled: bool = Field(True, description="Enable circuit breaker")
    failure_threshold: int = Field(5, ge=1, description="Failures to open circuit")
    recovery_timeout: int = Field(60, ge=10, description="Recovery timeout (seconds)")
    
    # Session affinity
    session_affinity: bool = Field(False, description="Enable session affinity")
    affinity_ttl: int = Field(3600, ge=60, description="Affinity TTL (seconds)")


class Backend:
    """Backend instance."""
    
    def __init__(
        self,
        id: str,
        region: str,
        endpoint: str,
        weight: int = 1
    ):
        """Initialize backend."""
        self.id = id
        self.region = region
        self.endpoint = endpoint
        self.weight = weight
        
        # Health
        self.healthy = True
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        
        # Metrics
        self.total_requests = 0
        self.active_connections = 0
        self.avg_response_time_ms = 0.0
        
        # Timing
        self.last_health_check = datetime.utcnow()


class LoadBalancer:
    """
    Production-grade load balancer.
    
    Features:
    - Multiple algorithms
    - Health-based routing
    - Session affinity
    - Automatic backend management
    """
    
    def __init__(self, config: HAConfig):
        """
        Initialize load balancer.
        
        Args:
            config: HA configuration
        """
        self.config = config
        self.backends: List[Backend] = []
        self.current_index = 0
        
        # Session affinity
        self.session_map: Dict[str, str] = {}  # session_id -> backend_id
        
        self.logger = StructuredLogger("load_balancer", LogLevel.INFO)
    
    def add_backend(self, backend: Backend):
        """Add a backend."""
        self.backends.append(backend)
        self.logger.info(
            "Backend added",
            backend_id=backend.id,
            region=backend.region,
            endpoint=backend.endpoint
        )
    
    def remove_backend(self, backend_id: str):
        """Remove a backend."""
        self.backends = [b for b in self.backends if b.id != backend_id]
        self.logger.info("Backend removed", backend_id=backend_id)
    
    def get_healthy_backends(self) -> List[Backend]:
        """Get healthy backends."""
        return [b for b in self.backends if b.healthy]
    
    async def select_backend(
        self,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None
    ) -> Optional[Backend]:
        """
        Select a backend using configured algorithm.
        
        Args:
            session_id: Session ID for affinity
            client_ip: Client IP for hash-based routing
        
        Returns:
            Selected backend or None
        """
        healthy_backends = self.get_healthy_backends()
        
        if not healthy_backends:
            self.logger.error("No healthy backends available")
            return None
        
        # Check session affinity
        if session_id and self.config.session_affinity:
            backend_id = self.session_map.get(session_id)
            if backend_id:
                backend = next((b for b in healthy_backends if b.id == backend_id), None)
                if backend:
                    return backend
        
        # Select based on algorithm
        if self.config.load_balancing_algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            backend = self._round_robin(healthy_backends)
        
        elif self.config.load_balancing_algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            backend = self._least_connections(healthy_backends)
        
        elif self.config.load_balancing_algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            backend = self._weighted_round_robin(healthy_backends)
        
        elif self.config.load_balancing_algorithm == LoadBalancingAlgorithm.IP_HASH:
            backend = self._ip_hash(healthy_backends, client_ip or "")
        
        elif self.config.load_balancing_algorithm == LoadBalancingAlgorithm.RANDOM:
            backend = random.choice(healthy_backends)
        
        elif self.config.load_balancing_algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            backend = self._least_response_time(healthy_backends)
        
        else:
            backend = healthy_backends[0]
        
        # Update session affinity
        if session_id and self.config.session_affinity and backend:
            self.session_map[session_id] = backend.id
        
        return backend
    
    def _round_robin(self, backends: List[Backend]) -> Backend:
        """Round robin selection."""
        backend = backends[self.current_index % len(backends)]
        self.current_index += 1
        return backend
    
    def _least_connections(self, backends: List[Backend]) -> Backend:
        """Least connections selection."""
        return min(backends, key=lambda b: b.active_connections)
    
    def _weighted_round_robin(self, backends: List[Backend]) -> Backend:
        """Weighted round robin selection."""
        # Build weighted list
        weighted_backends = []
        for backend in backends:
            weighted_backends.extend([backend] * backend.weight)
        
        if not weighted_backends:
            return backends[0]
        
        backend = weighted_backends[self.current_index % len(weighted_backends)]
        self.current_index += 1
        return backend
    
    def _ip_hash(self, backends: List[Backend], client_ip: str) -> Backend:
        """IP hash selection."""
        hash_value = hash(client_ip)
        index = hash_value % len(backends)
        return backends[index]
    
    def _least_response_time(self, backends: List[Backend]) -> Backend:
        """Least response time selection."""
        return min(backends, key=lambda b: b.avg_response_time_ms)
    
    async def record_request(
        self,
        backend_id: str,
        success: bool,
        response_time_ms: float
    ):
        """
        Record request result.
        
        Args:
            backend_id: Backend ID
            success: Request success
            response_time_ms: Response time
        """
        backend = next((b for b in self.backends if b.id == backend_id), None)
        if not backend:
            return
        
        backend.total_requests += 1
        
        # Update response time (exponential moving average)
        alpha = 0.2
        backend.avg_response_time_ms = (
            alpha * response_time_ms +
            (1 - alpha) * backend.avg_response_time_ms
        )
        
        # Update health
        if success:
            backend.consecutive_successes += 1
            backend.consecutive_failures = 0
            
            if backend.consecutive_successes >= self.config.healthy_threshold:
                if not backend.healthy:
                    backend.healthy = True
                    self.logger.info("Backend marked healthy", backend_id=backend_id)
        
        else:
            backend.consecutive_failures += 1
            backend.consecutive_successes = 0
            
            if backend.consecutive_failures >= self.config.unhealthy_threshold:
                if backend.healthy:
                    backend.healthy = False
                    self.logger.warning("Backend marked unhealthy", backend_id=backend_id)
        
        # Record metrics
        get_metrics().increment_counter(
            'lb_requests',
            {'backend_id': backend_id, 'success': str(success)},
            1
        )


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    Features:
    - Automatic failure detection
    - Fast fail mode
    - Gradual recovery
    - Per-service isolation
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_requests: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Failures to open circuit
            recovery_timeout: Time before attempting recovery
            half_open_requests: Test requests in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        # State
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.opened_at: Optional[datetime] = None
        
        self.logger = StructuredLogger(f"circuit_breaker_{name}", LogLevel.INFO)
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if self.opened_at:
                elapsed = (datetime.utcnow() - self.opened_at).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._transition_to_half_open()
                    return False
            return True
        return False
    
    def record_success(self):
        """Record successful request."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.half_open_requests:
                self._transition_to_closed()
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed request."""
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._transition_to_open()
        
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count += 1
            
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        self.state = CircuitBreakerState.OPEN
        self.opened_at = datetime.utcnow()
        self.success_count = 0
        
        self.logger.warning(
            "Circuit opened",
            failure_count=self.failure_count
        )
        
        get_metrics().increment_counter(
            'circuit_breaker_state_change',
            {'breaker': self.name, 'state': 'open'},
            1
        )
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.failure_count = 0
        
        self.logger.info("Circuit half-open, testing recovery")
        
        get_metrics().increment_counter(
            'circuit_breaker_state_change',
            {'breaker': self.name, 'state': 'half_open'},
            1
        )
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.opened_at = None
        
        self.logger.info("Circuit closed, normal operation resumed")
        
        get_metrics().increment_counter(
            'circuit_breaker_state_change',
            {'breaker': self.name, 'state': 'closed'},
            1
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None
        }


class FailoverManager:
    """
    Failover manager for automatic recovery.
    
    Features:
    - Automatic failover
    - Health monitoring
    - Region failover
    - Graceful degradation
    """
    
    def __init__(self, config: HAConfig):
        """
        Initialize failover manager.
        
        Args:
            config: HA configuration
        """
        self.config = config
        self.primary_region: Optional[str] = None
        self.active_regions: List[str] = []
        
        self.logger = StructuredLogger("failover_manager", LogLevel.INFO)
    
    def set_primary_region(self, region: str):
        """Set primary region."""
        if region not in self.config.regions:
            raise ValueError(f"Region {region} not in configuration")
        
        self.primary_region = region
        self.active_regions = [region]
        
        self.logger.info("Primary region set", region=region)
    
    async def check_region_health(self, region: str) -> bool:
        """
        Check region health.
        
        Args:
            region: Region to check
        
        Returns:
            True if healthy
        """
        # In production, this would check actual health endpoints
        # For now, return True as a placeholder
        return True
    
    async def failover_to_region(self, region: str):
        """
        Failover to a different region.
        
        Args:
            region: Target region
        """
        if region not in self.config.regions:
            raise ValueError(f"Region {region} not in configuration")
        
        self.logger.warning(
            "Initiating failover",
            from_region=self.primary_region,
            to_region=region
        )
        
        # Update active regions
        if self.config.strategy == FailoverStrategy.ACTIVE_ACTIVE:
            if region not in self.active_regions:
                self.active_regions.append(region)
        
        elif self.config.strategy == FailoverStrategy.ACTIVE_PASSIVE:
            self.active_regions = [region]
        
        self.primary_region = region
        
        self.logger.info(
            "Failover completed",
            primary_region=self.primary_region,
            active_regions=self.active_regions
        )
        
        # Record metrics
        get_metrics().increment_counter(
            'failover_events',
            {'to_region': region},
            1
        )
    
    async def monitor_and_failover(self):
        """Monitor regions and failover if needed."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check primary region
                if self.primary_region:
                    is_healthy = await self.check_region_health(self.primary_region)
                    
                    if not is_healthy:
                        # Find healthy region
                        for region in self.config.regions:
                            if region != self.primary_region:
                                if await self.check_region_health(region):
                                    await self.failover_to_region(region)
                                    break
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Failover monitoring error: {e}")

