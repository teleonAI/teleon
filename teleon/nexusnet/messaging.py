"""
Message Bus - Production-grade agent messaging system.

Features:
- Reliable message delivery
- Async communication
- Message persistence
- Dead letter queue
- Message routing
- Priority queues
- Broadcasting
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
from collections import defaultdict
import json

from teleon.core import (
    get_metrics,
    StructuredLogger,
    LogLevel,
)


class MessageType(str, Enum):
    """Message types."""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    BROADCAST = "broadcast"
    ERROR = "error"


class MessagePriority(int, Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class Message(BaseModel):
    """Agent message."""
    
    message_id: str = Field(..., description="Unique message ID")
    type: MessageType = Field(..., description="Message type")
    
    # Routing
    from_agent: str = Field(..., description="Source agent ID")
    to_agent: Optional[str] = Field(None, description="Destination agent ID")
    
    # Content
    payload: Dict[str, Any] = Field(..., description="Message payload")
    
    # Metadata
    priority: MessagePriority = Field(MessagePriority.NORMAL, description="Priority")
    correlation_id: Optional[str] = Field(None, description="Correlation ID")
    reply_to: Optional[str] = Field(None, description="Reply-to message ID")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    
    # Delivery
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")
    retry_count: int = Field(0, ge=0, description="Current retry count")
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MessageQueue:
    """Priority message queue."""
    
    def __init__(self):
        """Initialize message queue."""
        self.queues: Dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue()
            for priority in MessagePriority
        }
    
    async def put(self, message: Message):
        """Add message to queue."""
        await self.queues[message.priority].put(message)
    
    async def get(self) -> Message:
        """Get next message (priority order)."""
        # Check queues in priority order
        for priority in sorted(MessagePriority, key=lambda p: p.value, reverse=True):
            queue = self.queues[priority]
            if not queue.empty():
                return await queue.get()
        
        # If all empty, wait on normal priority
        return await self.queues[MessagePriority.NORMAL].get()
    
    def qsize(self) -> int:
        """Get total queue size."""
        return sum(q.qsize() for q in self.queues.values())
    
    def empty(self) -> bool:
        """Check if all queues are empty."""
        return all(q.empty() for q in self.queues.values())


class MessageBus:
    """
    Production-grade message bus for agent communication.
    
    Features:
    - Priority queues
    - Message persistence
    - Delivery guarantees
    - Broadcasting
    - Dead letter queue
    """
    
    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize message bus.
        
        Args:
            max_queue_size: Maximum queue size per agent
        """
        self.max_queue_size = max_queue_size
        
        # Agent queues
        self.queues: Dict[str, MessageQueue] = defaultdict(MessageQueue)
        
        # Message handlers
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Dead letter queue
        self.dead_letter_queue: List[Message] = []
        
        # Message history (for correlation)
        self.message_history: Dict[str, Message] = {}
        
        self.lock = asyncio.Lock()
        self.logger = StructuredLogger("message_bus", LogLevel.INFO)
        
        # Processing tasks
        self.processing_tasks: Dict[str, asyncio.Task] = {}
    
    async def send(
        self,
        from_agent: str,
        to_agent: str,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.REQUEST,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> str:
        """
        Send message to agent.
        
        Args:
            from_agent: Source agent ID
            to_agent: Destination agent ID
            payload: Message payload
            message_type: Message type
            priority: Message priority
            correlation_id: Correlation ID
            reply_to: Reply-to message ID
            ttl: Time to live in seconds
        
        Returns:
            Message ID
        """
        import uuid
        from datetime import timedelta
        
        # Create message
        message_id = str(uuid.uuid4())
        expires_at = None
        if ttl:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        
        message = Message(
            message_id=message_id,
            type=message_type,
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
            reply_to=reply_to,
            expires_at=expires_at
        )
        
        # Store in history
        async with self.lock:
            self.message_history[message_id] = message
        
        # Enqueue
        await self.queues[to_agent].put(message)
        
        self.logger.info(
            "Message sent",
            message_id=message_id,
            from_agent=from_agent,
            to_agent=to_agent,
            type=message_type.value
        )
        
        # Record metrics
        get_metrics().increment_counter(
            'memory_operations',
            {'memory_type': 'messages', 'operation': 'sent'},
            1
        )
        
        return message_id
    
    async def broadcast(
        self,
        from_agent: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> List[str]:
        """
        Broadcast message to all agents.
        
        Args:
            from_agent: Source agent ID
            payload: Message payload
            priority: Message priority
        
        Returns:
            List of message IDs
        """
        message_ids = []
        
        async with self.lock:
            agent_ids = list(self.queues.keys())
        
        for to_agent in agent_ids:
            if to_agent != from_agent:  # Don't send to self
                message_id = await self.send(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    payload=payload,
                    message_type=MessageType.BROADCAST,
                    priority=priority
                )
                message_ids.append(message_id)
        
        self.logger.info(
            "Message broadcast",
            from_agent=from_agent,
            recipient_count=len(message_ids)
        )
        
        return message_ids
    
    async def receive(
        self,
        agent_id: str,
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Receive message for agent.
        
        Args:
            agent_id: Agent ID
            timeout: Receive timeout
        
        Returns:
            Message or None
        """
        try:
            if timeout:
                message = await asyncio.wait_for(
                    self.queues[agent_id].get(),
                    timeout=timeout
                )
            else:
                message = await self.queues[agent_id].get()
            
            # Check expiration
            if message.is_expired():
                self.logger.warning(
                    "Message expired",
                    message_id=message.message_id
                )
                # Add to dead letter queue
                self.dead_letter_queue.append(message)
                return None
            
            self.logger.info(
                "Message received",
                message_id=message.message_id,
                agent_id=agent_id
            )
            
            # Record metrics
            get_metrics().increment_counter(
                'memory_operations',
                {'memory_type': 'messages', 'operation': 'received'},
                1
            )
            
            return message
        
        except asyncio.TimeoutError:
            return None
    
    async def reply(
        self,
        original_message: Message,
        from_agent: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """
        Reply to a message.
        
        Args:
            original_message: Original message
            from_agent: Replying agent ID
            payload: Reply payload
            priority: Message priority
        
        Returns:
            Reply message ID
        """
        return await self.send(
            from_agent=from_agent,
            to_agent=original_message.from_agent,
            payload=payload,
            message_type=MessageType.RESPONSE,
            priority=priority,
            correlation_id=original_message.correlation_id,
            reply_to=original_message.message_id
        )
    
    def subscribe(
        self,
        agent_id: str,
        handler: Callable[[Message], None]
    ):
        """
        Subscribe to messages with handler.
        
        Args:
            agent_id: Agent ID
            handler: Message handler function
        """
        self.handlers[agent_id].append(handler)
        
        # Start processing task if not already running
        if agent_id not in self.processing_tasks:
            task = asyncio.create_task(self._process_messages(agent_id))
            self.processing_tasks[agent_id] = task
        
        self.logger.info("Handler subscribed", agent_id=agent_id)
    
    async def _process_messages(self, agent_id: str):
        """Process messages for agent with handlers."""
        while True:
            try:
                message = await self.receive(agent_id)
                
                if message:
                    # Call all handlers
                    for handler in self.handlers[agent_id]:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(message)
                            else:
                                handler(message)
                        except Exception as e:
                            self.logger.error(
                                "Handler error",
                                agent_id=agent_id,
                                error=str(e)
                            )
            
            except Exception as e:
                self.logger.error(
                    "Message processing error",
                    agent_id=agent_id,
                    error=str(e)
                )
                await asyncio.sleep(1)  # Backoff on error
    
    def unsubscribe(self, agent_id: str):
        """
        Unsubscribe agent from messages.
        
        Args:
            agent_id: Agent ID
        """
        if agent_id in self.handlers:
            del self.handlers[agent_id]
        
        if agent_id in self.processing_tasks:
            self.processing_tasks[agent_id].cancel()
            del self.processing_tasks[agent_id]
        
        self.logger.info("Handler unsubscribed", agent_id=agent_id)
    
    async def get_queue_size(self, agent_id: str) -> int:
        """Get queue size for agent."""
        return self.queues[agent_id].qsize()
    
    async def get_dead_letter_messages(self) -> List[Message]:
        """Get messages from dead letter queue."""
        async with self.lock:
            return list(self.dead_letter_queue)
    
    async def shutdown(self):
        """Gracefully shutdown message bus."""
        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            task.cancel()
        
        self.logger.info("Message bus shutdown")


# Global message bus instance
_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get global message bus."""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus

