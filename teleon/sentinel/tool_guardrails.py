"""
Tool Call Guardrails.

Wraps tool functions to enforce Sentinel policies before each invocation.
"""

import asyncio
import functools
from typing import Any, Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from teleon.sentinel.engine import SentinelEngine


class ToolGuardrail:
    """
    Wraps tool functions with Sentinel validation.

    On violation:
    - BLOCK: raises AgentValidationError
    - FLAG: logs the violation, continues execution
    - ESCALATE: marks for review, continues execution
    """

    def __init__(self, engine: "SentinelEngine"):
        self.engine = engine

    def wrap_tool(self, tool_func: Callable, tool_name: Optional[str] = None) -> Callable:
        """Wrap a single tool function with guardrail checks."""
        name = tool_name or getattr(tool_func, "__name__", "unknown_tool")

        if asyncio.iscoroutinefunction(tool_func):
            @functools.wraps(tool_func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                await self.engine.validate_tool_call(
                    tool_name=name,
                    tool_args=kwargs if kwargs else {"args": args},
                    agent_name=self.engine.config.language,  # Will be overridden by caller context
                )
                return await tool_func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(tool_func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass

                if loop and loop.is_running():
                    # We're in an async context but calling a sync tool.
                    # Create a future for the validation and run it.
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(
                            asyncio.run,
                            self.engine.validate_tool_call(
                                tool_name=name,
                                tool_args=kwargs if kwargs else {"args": args},
                                agent_name="",
                            ),
                        )
                        future.result(timeout=5.0)
                else:
                    asyncio.run(
                        self.engine.validate_tool_call(
                            tool_name=name,
                            tool_args=kwargs if kwargs else {"args": args},
                            agent_name="",
                        )
                    )
                return tool_func(*args, **kwargs)
            return sync_wrapper

    def guard_tools(self, tools: List[Callable]) -> List[Callable]:
        """Wrap a list of tool functions with guardrail checks."""
        return [self.wrap_tool(t) for t in tools]
