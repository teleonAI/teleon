"""Communication tools."""

from typing import Any
from teleon.tools.base import BaseTool, ToolResult, ToolSchema, ToolCategory


class SendEmailTool(BaseTool):
    """Send email."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Send email."""
        to = kwargs.get("to")
        subject = kwargs.get("subject")
        body = kwargs.get("body")
        from_addr = kwargs.get("from", "noreply@teleon.ai")
        
        try:
            # Placeholder - in production, use SMTP or email service
            return ToolResult(
                success=True,
                data={"sent": True, "to": to, "subject": subject},
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="send_email",
            description="Send an email",
            category=ToolCategory.COMMUNICATION,
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "from": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            },
            returns={"type": "object"},
            tags=["email", "send", "communication"]
        )


class SendSMSTool(BaseTool):
    """Send SMS."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Send SMS."""
        to = kwargs.get("to")
        message = kwargs.get("message")
        
        try:
            # Placeholder - in production, use Twilio or similar
            return ToolResult(
                success=True,
                data={"sent": True, "to": to},
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="send_sms",
            description="Send SMS message",
            category=ToolCategory.COMMUNICATION,
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "message": {"type": "string"}
                },
                "required": ["to", "message"]
            },
            returns={"type": "object"},
            tags=["sms", "text", "communication"]
        )


class SlackMessageTool(BaseTool):
    """Send Slack message."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Send Slack message."""
        channel = kwargs.get("channel")
        text = kwargs.get("text")
        
        try:
            # Placeholder - in production, use Slack API
            return ToolResult(
                success=True,
                data={"sent": True, "channel": channel},
                tool_name=self.name
            )
        
        except Exception as e:
            return ToolResult(success=False, error=str(e), tool_name=self.name)
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="slack_message",
            description="Send message to Slack channel",
            category=ToolCategory.COMMUNICATION,
            parameters={
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "text": {"type": "string"}
                },
                "required": ["channel", "text"]
            },
            returns={"type": "object"},
            tags=["slack", "message", "communication"]
        )

