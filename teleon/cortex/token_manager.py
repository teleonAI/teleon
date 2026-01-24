"""
Token Manager - Accurate token counting and management for LLM operations.

Provides utilities for:
- Accurate token counting for various models
- Context window management
- Token budget tracking
- Text truncation to token limits
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Accurate token counting for various LLM models.
    
    Supports:
    - GPT-4, GPT-3.5, and other OpenAI models
    - Claude models
    - Custom models
    
    Example:
        ```python
        counter = TokenCounter(model="gpt-4")
        tokens = counter.count_tokens("Hello, world!")
        
        # Count message tokens (OpenAI format)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        total = counter.count_messages(messages)
        ```
    """
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize token counter.
        
        Args:
            model: Model name for accurate counting
        """
        self.model = model
        self._encoding = None
        self._initialize_encoding()
    
    def _initialize_encoding(self):
        """Initialize tokenizer encoding."""
        try:
            import tiktoken
            try:
                self._encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                logger.warning(f"Model {self.model} not found, using cl100k_base encoding")
                self._encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not installed, using approximate counting")
            self._encoding = None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self._encoding:
            return len(self._encoding.encode(text))
        else:
            # Approximate: ~4 characters per token for English
            return len(text) // 4
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in message list (OpenAI format).
        
        Accounts for message formatting overhead.
        
        Args:
            messages: List of messages with 'role' and 'content'
        
        Returns:
            Total token count including formatting
        """
        if not messages:
            return 0
        
        total_tokens = 0
        
        # Model-specific token overhead
        if self.model.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
        elif self.model.startswith("gpt-3.5"):
            tokens_per_message = 4
            tokens_per_name = -1
        else:
            tokens_per_message = 3
            tokens_per_name = 1
        
        for message in messages:
            total_tokens += tokens_per_message
            
            for key, value in message.items():
                total_tokens += self.count_tokens(str(value))
                if key == "name":
                    total_tokens += tokens_per_name
        
        # Every reply is primed with assistant
        total_tokens += 3
        
        return total_tokens
    
    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
        from_end: bool = False
    ) -> str:
        """
        Truncate text to fit token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            from_end: If True, keep end of text; if False, keep beginning
        
        Returns:
            Truncated text
        """
        if not text:
            return text
        
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        if not self._encoding:
            # Approximate truncation
            chars_per_token = len(text) // current_tokens
            max_chars = max_tokens * chars_per_token
            
            if from_end:
                return text[-max_chars:]
            else:
                return text[:max_chars]
        
        # Encode and truncate
        tokens = self._encoding.encode(text)
        
        if from_end:
            truncated_tokens = tokens[-max_tokens:]
        else:
            truncated_tokens = tokens[:max_tokens]
        
        return self._encoding.decode(truncated_tokens)
    
    def estimate_response_tokens(
        self,
        prompt_tokens: int,
        model: Optional[str] = None
    ) -> int:
        """
        Estimate response tokens based on model and prompt.
        
        Rules of thumb:
        - GPT-4: ~50% of prompt length
        - GPT-3.5: ~30% of prompt length
        - Claude: ~60% of prompt length
        
        Args:
            prompt_tokens: Number of tokens in prompt
            model: Model name (uses self.model if not provided)
        
        Returns:
            Estimated response tokens
        """
        model = model or self.model
        
        if "gpt-4" in model.lower():
            return int(prompt_tokens * 0.5)
        elif "gpt-3.5" in model.lower():
            return int(prompt_tokens * 0.3)
        elif "claude" in model.lower():
            return int(prompt_tokens * 0.6)
        else:
            return int(prompt_tokens * 0.4)
    
    def split_by_tokens(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0
    ) -> List[str]:
        """
        Split text into chunks by token count.
        
        Args:
            text: Text to split
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
        
        Returns:
            List of text chunks
        """
        if not text or not self._encoding:
            # Fallback to character-based splitting
            chars_per_token = 4
            char_chunk_size = chunk_size * chars_per_token
            char_overlap = overlap * chars_per_token
            
            chunks = []
            start = 0
            while start < len(text):
                end = start + char_chunk_size
                chunks.append(text[start:end])
                start = end - char_overlap if overlap > 0 else end
            
            return chunks
        
        tokens = self._encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(self._encoding.decode(chunk_tokens))
            
            # Move start position with overlap
            start = end - overlap if overlap > 0 else end
        
        return chunks


class ContextWindowManager:
    """
    Manage context window limits for different models.
    
    Tracks:
    - Maximum context window size
    - Used tokens
    - Available tokens
    - Warnings when approaching limits
    
    Example:
        ```python
        manager = ContextWindowManager(model="gpt-4", max_tokens=8192)
        
        # Add tokens
        manager.add_tokens("system", 100)
        manager.add_tokens("user_prompt", 500)
        
        # Check availability
        available = manager.get_available_tokens()
        
        # Get usage report
        report = manager.get_usage_report()
        ```
    """
    
    # Model context windows
    MODEL_LIMITS = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16385,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-2": 100000,
    }
    
    def __init__(self, model: str = "gpt-4", max_tokens: Optional[int] = None):
        """
        Initialize context window manager.
        
        Args:
            model: Model name
            max_tokens: Override default max tokens
        """
        self.model = model
        self.max_tokens = max_tokens or self._get_model_limit(model)
        self.token_counter = TokenCounter(model)
        
        # Track token usage by category
        self.token_usage: Dict[str, int] = {}
        self.total_used = 0
    
    def _get_model_limit(self, model: str) -> int:
        """Get context window limit for model."""
        for key, limit in self.MODEL_LIMITS.items():
            if key in model.lower():
                return limit
        
        # Default to conservative limit
        logger.warning(f"Unknown model {model}, using default 4096 token limit")
        return 4096
    
    def add_tokens(self, category: str, tokens: int):
        """
        Add tokens to usage tracking.
        
        Args:
            category: Category name (e.g., 'system', 'context', 'user')
            tokens: Number of tokens
        """
        self.token_usage[category] = self.token_usage.get(category, 0) + tokens
        self.total_used += tokens
        
        # Warn if approaching limit
        if self.total_used > self.max_tokens * 0.9:
            logger.warning(
                f"Context window at {self.get_usage_percentage():.1f}% capacity"
            )
    
    def add_text(self, category: str, text: str):
        """
        Add text and automatically count tokens.
        
        Args:
            category: Category name
            text: Text content
        """
        tokens = self.token_counter.count_tokens(text)
        self.add_tokens(category, tokens)
    
    def get_available_tokens(self) -> int:
        """
        Get number of available tokens.
        
        Returns:
            Available tokens
        """
        return max(0, self.max_tokens - self.total_used)
    
    def get_usage_percentage(self) -> float:
        """
        Get usage percentage.
        
        Returns:
            Percentage of context window used (0-100)
        """
        return (self.total_used / self.max_tokens) * 100
    
    def can_fit(self, tokens: int) -> bool:
        """
        Check if tokens can fit in remaining space.
        
        Args:
            tokens: Number of tokens to check
        
        Returns:
            True if tokens fit
        """
        return self.total_used + tokens <= self.max_tokens
    
    def get_usage_report(self) -> Dict[str, Any]:
        """
        Get detailed usage report.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "total_used": self.total_used,
            "available": self.get_available_tokens(),
            "usage_percentage": round(self.get_usage_percentage(), 2),
            "by_category": dict(self.token_usage),
            "is_near_limit": self.total_used > self.max_tokens * 0.9
        }
    
    def reset(self):
        """Reset token usage tracking."""
        self.token_usage.clear()
        self.total_used = 0


class TokenBudgetTracker:
    """
    Track token usage and costs across multiple requests.
    
    Features:
    - Track cumulative token usage
    - Calculate costs
    - Budget enforcement
    - Usage analytics
    
    Example:
        ```python
        tracker = TokenBudgetTracker(
            budget_tokens=1000000,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03
        )
        
        # Record usage
        tracker.record_usage(
            input_tokens=500,
            output_tokens=200,
            metadata={'agent': 'agent-123'}
        )
        
        # Check budget
        if tracker.is_budget_exceeded():
            print("Budget exceeded!")
        
        # Get report
        report = tracker.get_report()
        ```
    """
    
    def __init__(
        self,
        budget_tokens: Optional[int] = None,
        cost_per_1k_input: float = 0.01,
        cost_per_1k_output: float = 0.03
    ):
        """
        Initialize budget tracker.
        
        Args:
            budget_tokens: Maximum tokens allowed (None = unlimited)
            cost_per_1k_input: Cost per 1K input tokens
            cost_per_1k_output: Cost per 1K output tokens
        """
        self.budget_tokens = budget_tokens
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        
        # Tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.request_history: List[Dict[str, Any]] = []
    
    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Optional metadata about the request
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        
        # Calculate cost for this request
        cost = self.calculate_cost(input_tokens, output_tokens)
        
        # Record in history
        self.request_history.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost,
            "metadata": metadata or {}
        })
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for tokens.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Cost in dollars
        """
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost
    
    def get_total_tokens(self) -> int:
        """Get total tokens used."""
        return self.total_input_tokens + self.total_output_tokens
    
    def get_total_cost(self) -> float:
        """Get total cost."""
        return self.calculate_cost(self.total_input_tokens, self.total_output_tokens)
    
    def is_budget_exceeded(self) -> bool:
        """Check if budget is exceeded."""
        if self.budget_tokens is None:
            return False
        return self.get_total_tokens() > self.budget_tokens
    
    def get_remaining_budget(self) -> Optional[int]:
        """Get remaining token budget."""
        if self.budget_tokens is None:
            return None
        return max(0, self.budget_tokens - self.get_total_tokens())
    
    def get_report(self) -> Dict[str, Any]:
        """
        Get usage report.
        
        Returns:
            Dictionary with usage statistics
        """
        total_tokens = self.get_total_tokens()
        total_cost = self.get_total_cost()
        
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
            "avg_tokens_per_request": round(total_tokens / max(self.total_requests, 1), 2),
            "avg_cost_per_request": round(total_cost / max(self.total_requests, 1), 4),
            "budget_tokens": self.budget_tokens,
            "remaining_budget": self.get_remaining_budget(),
            "budget_exceeded": self.is_budget_exceeded()
        }
    
    def reset(self):
        """Reset all tracking."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.request_history.clear()


__all__ = [
    "TokenCounter",
    "ContextWindowManager",
    "TokenBudgetTracker",
]

