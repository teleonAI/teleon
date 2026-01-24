"""
Batch Processing for LLM Requests.

This module provides intelligent request batching to improve efficiency
and reduce costs for LLM operations.
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field, ConfigDict, field_serializer
from datetime import datetime, timezone
import asyncio
import time

from teleon.core import (
    StructuredLogger,
    LogLevel,
)


class BatchRequest(BaseModel):
    """A batched request."""
    
    request_id: str = Field(..., description="Request identifier")
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    model: str = Field(..., description="Model name")
    max_tokens: Optional[int] = Field(None, description="Max tokens")
    temperature: float = Field(0.7, description="Temperature")
    
    # Metadata
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: Optional[str] = Field(None, description="Agent ID")
    priority: int = Field(0, description="Priority (higher = more urgent)")

    # Callback for result
    result_future: Any = Field(None, description="Future for result", exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer('submitted_at')
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat() if value else None


class BatchConfig(BaseModel):
    """Configuration for batch processing."""
    
    max_batch_size: int = Field(10, ge=1, le=100, description="Max requests per batch")
    max_wait_time_ms: float = Field(1000.0, gt=0, description="Max wait time before processing")
    min_batch_size: int = Field(2, ge=1, description="Min size to process as batch")
    
    # Priority handling
    enable_priority: bool = Field(True, description="Enable priority-based batching")
    priority_boost_threshold: int = Field(5, description="Priority level that bypasses batching")


class Batch(BaseModel):
    """A collection of requests to process together."""
    
    batch_id: str = Field(..., description="Batch identifier")
    requests: List[BatchRequest] = Field(default_factory=list, description="Batched requests")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model: str = Field(..., description="Model for this batch")
    total_tokens: int = Field(0, description="Estimated total tokens")

    model_config = ConfigDict()

    @field_serializer('created_at')
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat() if value else None
    
    def can_add(self, request: BatchRequest, max_size: int) -> bool:
        """Check if request can be added to batch."""
        if len(self.requests) >= max_size:
            return False
        
        # Must be same model
        if request.model != self.model:
            return False
        
        return True
    
    def add_request(self, request: BatchRequest):
        """Add request to batch."""
        self.requests.append(request)


class RequestBatcher:
    """
    Group similar requests for batch processing.
    
    Features:
    - Group by model
    - Respect max batch size and wait time
    - Priority-based flushing
    - Automatic batch optimization
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize request batcher.
        
        Args:
            config: Batch configuration
        """
        self.config = config or BatchConfig()
        
        # Pending requests by model
        self.pending_by_model: Dict[str, List[BatchRequest]] = {}
        
        # Active batches
        self.active_batches: List[Batch] = []
        
        # Batch processor callback
        self.batch_processor: Optional[Callable] = None
        
        # Statistics
        self.total_requests = 0
        self.total_batches = 0
        self.total_batched_requests = 0
        
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger("request_batcher", LogLevel.INFO)
        
        # Background task for auto-flushing
        self.flush_task: Optional[asyncio.Task] = None
        self.running = False
    
    def set_batch_processor(self, processor: Callable):
        """Set the batch processor callback."""
        self.batch_processor = processor
    
    async def submit_request(
        self,
        messages: List[Dict[str, str]],
        model: str,
        agent_id: Optional[str] = None,
        priority: int = 0,
        **kwargs
    ) -> Any:
        """
        Submit a request for batching.
        
        Args:
            messages: Chat messages
            model: Model name
            agent_id: Agent ID
            priority: Request priority
            **kwargs: Additional request parameters
        
        Returns:
            Result future
        """
        import uuid
        
        # Create request
        result_future = asyncio.Future()
        
        request = BatchRequest(
            request_id=str(uuid.uuid4()),
            messages=messages,
            model=model,
            agent_id=agent_id,
            priority=priority,
            result_future=result_future,
            **{k: v for k, v in kwargs.items() if k in BatchRequest.__fields__}
        )
        
        async with self.lock:
            self.total_requests += 1
            
            # Check if priority bypass
            if self.config.enable_priority and priority >= self.config.priority_boost_threshold:
                # Process immediately without batching
                self.logger.debug(f"High priority request {request.request_id} - processing immediately")
                # Schedule for immediate processing
                asyncio.create_task(self._process_single(request))
                return result_future
            
            # Add to pending
            if model not in self.pending_by_model:
                self.pending_by_model[model] = []
            
            self.pending_by_model[model].append(request)
            
            # Check if we should flush this model's batch
            if len(self.pending_by_model[model]) >= self.config.max_batch_size:
                await self._flush_model(model)
        
        return result_future
    
    async def _process_single(self, request: BatchRequest):
        """Process a single request without batching."""
        if not self.batch_processor:
            request.result_future.set_exception(
                RuntimeError("No batch processor configured")
            )
            return
        
        try:
            result = await self.batch_processor([request])
            if result and len(result) > 0:
                request.result_future.set_result(result[0])
            else:
                request.result_future.set_exception(
                    RuntimeError("No result from processor")
                )
        except Exception as e:
            request.result_future.set_exception(e)
    
    async def _flush_model(self, model: str):
        """Flush pending requests for a model."""
        if model not in self.pending_by_model:
            return
        
        pending = self.pending_by_model[model]
        if not pending:
            return
        
        # Take up to max_batch_size requests
        batch_requests = pending[:self.config.max_batch_size]
        self.pending_by_model[model] = pending[self.config.max_batch_size:]
        
        if len(batch_requests) < self.config.min_batch_size:
            # Not enough for a batch, process individually
            for req in batch_requests:
                asyncio.create_task(self._process_single(req))
            return
        
        # Create batch
        import uuid
        batch = Batch(
            batch_id=str(uuid.uuid4()),
            requests=batch_requests,
            model=model
        )
        
        self.active_batches.append(batch)
        self.total_batches += 1
        self.total_batched_requests += len(batch_requests)
        
        self.logger.info(
            f"Created batch {batch.batch_id} with {len(batch_requests)} requests for {model}"
        )
        
        # Process batch
        asyncio.create_task(self._process_batch(batch))
    
    async def _process_batch(self, batch: Batch):
        """Process a batch of requests."""
        if not self.batch_processor:
            for req in batch.requests:
                req.result_future.set_exception(
                    RuntimeError("No batch processor configured")
                )
            return
        
        try:
            start_time = time.time()
            
            # Call batch processor with timeout (default: 60 seconds)
            batch_timeout = 60.0  # seconds
            try:
                results = await asyncio.wait_for(
                    self.batch_processor(batch.requests),
                    timeout=batch_timeout
                )
            except asyncio.TimeoutError:
                error = TimeoutError(
                    f"Batch processing timed out after {batch_timeout}s"
                )
                for req in batch.requests:
                    req.result_future.set_exception(error)
                self.logger.error(
                    f"Batch {batch.batch_id} processing timed out"
                )
                return
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Distribute results
            if len(results) == len(batch.requests):
                for req, result in zip(batch.requests, results):
                    req.result_future.set_result(result)
            else:
                # Mismatch - set errors
                error = RuntimeError(
                    f"Batch processor returned {len(results)} results "
                    f"for {len(batch.requests)} requests"
                )
                for req in batch.requests:
                    req.result_future.set_exception(error)
            
            self.logger.info(
                f"Processed batch {batch.batch_id} in {duration_ms:.1f}ms"
            )
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            for req in batch.requests:
                req.result_future.set_exception(e)
        
        finally:
            # Remove from active batches
            async with self.lock:
                if batch in self.active_batches:
                    self.active_batches.remove(batch)
    
    async def start(self):
        """Start background flushing."""
        if self.running:
            return
        
        self.running = True
        self.flush_task = asyncio.create_task(self._flush_loop())
        self.logger.info("Request batcher started")
    
    async def stop(self):
        """Stop background flushing."""
        self.running = False
        
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush all pending
        await self.flush_all()
        
        self.logger.info("Request batcher stopped")
    
    async def _flush_loop(self):
        """Background loop to flush batches periodically."""
        while self.running:
            try:
                # Wait for max_wait_time
                await asyncio.sleep(self.config.max_wait_time_ms / 1000.0)
                
                # Flush all models with pending requests
                async with self.lock:
                    models_to_flush = list(self.pending_by_model.keys())
                
                for model in models_to_flush:
                    await self._flush_model(model)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Flush loop error: {e}")
    
    async def flush_all(self):
        """Flush all pending requests."""
        async with self.lock:
            models = list(self.pending_by_model.keys())
        
        for model in models:
            await self._flush_model(model)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batching statistics."""
        batching_efficiency = (
            self.total_batched_requests / max(self.total_requests, 1)
        )
        
        avg_batch_size = (
            self.total_batched_requests / max(self.total_batches, 1)
        )
        
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "total_batched_requests": self.total_batched_requests,
            "batching_efficiency": batching_efficiency,
            "avg_batch_size": avg_batch_size,
            "active_batches": len(self.active_batches),
            "pending_requests": sum(
                len(reqs) for reqs in self.pending_by_model.values()
            )
        }


class BatchOptimizer:
    """
    Optimize batch sizes and timing.
    
    Analyzes performance and cost to recommend optimal batch configuration.
    """
    
    def __init__(self, batcher: RequestBatcher):
        """
        Initialize batch optimizer.
        
        Args:
            batcher: Request batcher to optimize
        """
        self.batcher = batcher
        self.logger = StructuredLogger("batch_optimizer", LogLevel.INFO)
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze batching performance.
        
        Returns:
            Performance analysis
        """
        stats = self.batcher.get_statistics()
        
        analysis = {
            "efficiency": stats["batching_efficiency"],
            "avg_batch_size": stats["avg_batch_size"],
            "recommendations": []
        }
        
        # Analyze efficiency
        if stats["batching_efficiency"] < 0.5:
            analysis["recommendations"].append({
                "type": "configuration",
                "priority": "high",
                "message": "Low batching efficiency (<50%). Consider increasing max_wait_time_ms."
            })
        
        # Analyze batch size
        if stats["avg_batch_size"] < 3:
            analysis["recommendations"].append({
                "type": "configuration",
                "priority": "medium",
                "message": "Small average batch size. Consider increasing max_wait_time_ms or max_batch_size."
            })
        
        # Check pending queue
        if stats["pending_requests"] > 20:
            analysis["recommendations"].append({
                "type": "capacity",
                "priority": "high",
                "message": f"High pending queue ({stats['pending_requests']} requests). Consider increasing processing capacity."
            })
        
        return analysis
    
    def recommend_config(self) -> BatchConfig:
        """
        Recommend optimal batch configuration.
        
        Returns:
            Recommended configuration
        """
        stats = self.batcher.get_statistics()
        
        # Start with current config
        new_config = self.batcher.config.copy()
        
        # Adjust based on performance
        if stats["batching_efficiency"] < 0.5 and stats["avg_batch_size"] < 5:
            # Increase wait time to allow larger batches
            new_config.max_wait_time_ms = min(
                new_config.max_wait_time_ms * 1.5,
                3000.0
            )
        
        if stats["avg_batch_size"] > 8:
            # Batches are large, can reduce wait time
            new_config.max_wait_time_ms = max(
                new_config.max_wait_time_ms * 0.8,
                500.0
            )
        
        return new_config


class BatchProcessor:
    """
    High-level batch processor that integrates batching with LLM calls.
    
    Provides simple interface for batch-enabled LLM operations.
    """
    
    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        llm_caller: Optional[Callable] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            config: Batch configuration
            llm_caller: Function to call LLM (receives List[BatchRequest])
        """
        self.batcher = RequestBatcher(config)
        self.llm_caller = llm_caller
        
        # Set self as batch processor
        self.batcher.set_batch_processor(self._process_llm_batch)
        
        self.logger = StructuredLogger("batch_processor", LogLevel.INFO)
    
    async def _process_llm_batch(
        self,
        requests: List[BatchRequest]
    ) -> List[Any]:
        """Process a batch of LLM requests."""
        if not self.llm_caller:
            raise RuntimeError("No LLM caller configured")
        
        return await self.llm_caller(requests)
    
    async def process_request(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> Any:
        """
        Process a request (with batching).
        
        Args:
            messages: Chat messages
            model: Model name
            **kwargs: Additional parameters
        
        Returns:
            LLM response
        """
        result = await self.batcher.submit_request(
            messages=messages,
            model=model,
            **kwargs
        )
        
        return await result
    
    async def start(self):
        """Start batch processing."""
        await self.batcher.start()
    
    async def stop(self):
        """Stop batch processing."""
        await self.batcher.stop()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return self.batcher.get_statistics()


# Global batch processor
_global_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> BatchProcessor:
    """Get global batch processor."""
    global _global_batch_processor
    if _global_batch_processor is None:
        _global_batch_processor = BatchProcessor()
    return _global_batch_processor

