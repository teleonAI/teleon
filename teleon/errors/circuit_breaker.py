"""Circuit Breaker pattern implementation."""

import asyncio
from enum import Enum
from typing import Callable, TypeVar, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 60.0  # Seconds before trying again
    expected_exception: Type[Exception] = Exception


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.
    
    Prevents cascading failures by failing fast when a service is down.
    """
    
    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        name: str = "circuit"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
            name: Name for identification
        """
        self.config = config or CircuitBreakerConfig()
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    async def call(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerError: If circuit is open
        """
        async with self._lock:
            # Check if circuit should transition to half-open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    print(f"Circuit '{self.name}' transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN"
                    )
        
        # Execute the function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.config.expected_exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self) -> None:
        """Handle successful execution."""
        async with self._lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
                if self.success_count >= self.config.success_threshold:
                    print(f"Circuit '{self.name}' transitioning to CLOSED")
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
    
    async def _on_failure(self) -> None:
        """Handle failed execution."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.state == CircuitState.HALF_OPEN:
                print(f"Circuit '{self.name}' transitioning to OPEN (failed in half-open)")
                self.state = CircuitState.OPEN
                self.success_count = 0
            elif self.failure_count >= self.config.failure_threshold:
                print(f"Circuit '{self.name}' transitioning to OPEN ({self.failure_count} failures)")
                self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = datetime.utcnow() - self.last_failure_time
        return elapsed.total_seconds() >= self.config.timeout
    
    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            print(f"Circuit '{self.name}' manually reset to CLOSED")
    
    def get_state(self) -> dict:
        """
        Get current circuit breaker state.
        
        Returns:
            State information
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }

