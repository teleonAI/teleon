"""Retry logic with exponential backoff."""

import asyncio
from typing import Callable, TypeVar, Optional, Type
from dataclasses import dataclass
import time

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_retries: int = 3
    initial_delay: float = 0.5  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retry_on_exceptions: tuple = (Exception,)


class RetryError(Exception):
    """Raised when all retries are exhausted."""
    
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Failed after {attempts} attempts. Last error: {last_exception}"
        )


async def retry_async(
    func: Callable[..., T],
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs
) -> T:
    """
    Execute an async function with retry logic.
    
    Args:
        func: Async function to execute
        config: Retry configuration
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
    
    Returns:
        Function result
    
    Raises:
        RetryError: If all retries are exhausted
    """
    config = config or RetryConfig()
    last_exception = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except config.retry_on_exceptions as e:
            last_exception = e
            
            if attempt >= config.max_retries:
                raise RetryError(attempt + 1, last_exception)
            
            # Calculate delay with exponential backoff
            delay = min(
                config.initial_delay * (config.exponential_base ** attempt),
                config.max_delay
            )
            
            # Add jitter to prevent thundering herd
            if config.jitter:
                import random
                delay = delay * (0.5 + random.random())
            
            print(f"Retry attempt {attempt + 1}/{config.max_retries} after {delay:.2f}s delay")
            await asyncio.sleep(delay)
    
    raise RetryError(config.max_retries + 1, last_exception)

