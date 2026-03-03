"""Communication tools."""

import asyncio
import os
import smtplib
from email.message import EmailMessage
from typing import Any

from teleon.tools.base import BaseTool, ToolResult, ToolSchema, ToolCategory


class SendEmailTool(BaseTool):
    """Send email."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Send email."""
        to = kwargs.get("to")
        subject = kwargs.get("subject")
        body = kwargs.get("body")
        from_addr = kwargs.get("from") or os.getenv("TELEON_EMAIL_FROM") or "noreply@teleon.ai"
        cc = kwargs.get("cc")
        bcc = kwargs.get("bcc")
        reply_to = kwargs.get("reply_to")
        is_html = bool(kwargs.get("is_html", False))
        dry_run = bool(kwargs.get("dry_run", False))

        host = kwargs.get("smtp_host") or os.getenv("TELEON_SMTP_HOST")
        port = int(kwargs.get("smtp_port") or os.getenv("TELEON_SMTP_PORT") or 587)
        username = kwargs.get("smtp_username") or os.getenv("TELEON_SMTP_USERNAME")
        password = kwargs.get("smtp_password") or os.getenv("TELEON_SMTP_PASSWORD")
        use_tls = bool(kwargs.get("smtp_use_tls", os.getenv("TELEON_SMTP_USE_TLS", "true")).lower() != "false")
        use_ssl = bool(kwargs.get("smtp_use_ssl", os.getenv("TELEON_SMTP_USE_SSL", "false")).lower() == "true")
        
        try:
            if not host:
                return ToolResult(
                    success=False,
                    error=(
                        "Email tool not configured. Missing SMTP host. "
                        "Set TELEON_SMTP_HOST (and typically TELEON_SMTP_USERNAME/TELEON_SMTP_PASSWORD)."
                    ),
                    tool_name=self.name,
                    metadata={
                        "missing_env": ["TELEON_SMTP_HOST"],
                        "required_env": [
                            "TELEON_SMTP_HOST",
                            "TELEON_SMTP_PORT",
                            "TELEON_SMTP_USERNAME",
                            "TELEON_SMTP_PASSWORD",
                            "TELEON_EMAIL_FROM",
                        ],
                    },
                )

            msg = EmailMessage()
            msg["To"] = to
            msg["From"] = from_addr
            msg["Subject"] = subject

            if cc:
                msg["Cc"] = cc
            if reply_to:
                msg["Reply-To"] = reply_to

            recipients = [to]
            if cc:
                recipients.extend([x.strip() for x in str(cc).split(",") if x.strip()])
            if bcc:
                recipients.extend([x.strip() for x in str(bcc).split(",") if x.strip()])

            if is_html:
                msg.set_content(body)
                msg.add_alternative(body, subtype="html")
            else:
                msg.set_content(body)

            if dry_run:
                return ToolResult(
                    success=True,
                    data={
                        "sent": False,
                        "dry_run": True,
                        "to": to,
                        "subject": subject,
                        "from": from_addr,
                        "smtp": {"host": host, "port": port, "use_tls": use_tls, "use_ssl": use_ssl},
                    },
                    tool_name=self.name,
                )

            def _send() -> None:
                if use_ssl:
                    server: smtplib.SMTP = smtplib.SMTP_SSL(host=host, port=port, timeout=30)
                else:
                    server = smtplib.SMTP(host=host, port=port, timeout=30)
                try:
                    server.ehlo()
                    if use_tls and not use_ssl:
                        server.starttls()
                        server.ehlo()
                    if username and password:
                        server.login(username, password)
                    server.send_message(msg, from_addr=from_addr, to_addrs=recipients)
                finally:
                    try:
                        server.quit()
                    except Exception:
                        pass

            await asyncio.to_thread(_send)

            return ToolResult(
                success=True,
                data={"sent": True, "to": to, "subject": subject, "from": from_addr},
                tool_name=self.name,
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
                    "from": {"type": "string"},
                    "cc": {"type": "string"},
                    "bcc": {"type": "string"},
                    "reply_to": {"type": "string"},
                    "is_html": {"type": "boolean"},
                    "dry_run": {"type": "boolean"},
                    "smtp_host": {"type": "string"},
                    "smtp_port": {"type": "integer"},
                    "smtp_username": {"type": "string"},
                    "smtp_password": {"type": "string"},
                    "smtp_use_tls": {"type": "boolean"},
                    "smtp_use_ssl": {"type": "boolean"},
                },
                "required": ["to", "subject", "body"]
            },
            returns={"type": "object"},
            tags=["email", "send", "communication"],
            requires=[
                "SMTP server access",
                "Env: TELEON_SMTP_HOST",
            ],
            examples=[
                {
                    "to": "test@example.com",
                    "subject": "Hello",
                    "body": "Test email",
                    "dry_run": True,
                }
            ],
        )


class SendSMSTool(BaseTool):
    """Send SMS."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Send SMS."""
        to = kwargs.get("to")
        message = kwargs.get("message")
        from_number = kwargs.get("from") or os.getenv("TELEON_TWILIO_FROM")
        dry_run = bool(kwargs.get("dry_run", False))

        account_sid = kwargs.get("twilio_account_sid") or os.getenv("TELEON_TWILIO_ACCOUNT_SID")
        auth_token = kwargs.get("twilio_auth_token") or os.getenv("TELEON_TWILIO_AUTH_TOKEN")
        
        try:
            try:
                from twilio.rest import Client as TwilioClient
            except ImportError:
                return ToolResult(
                    success=False,
                    error=(
                        "Twilio dependency not installed. Install with: pip install teleon[twilio] "
                        "(or pip install twilio)."
                    ),
                    tool_name=self.name,
                    metadata={"missing_dependency": "twilio", "install": "teleon[twilio]"},
                )

            missing = []
            if not account_sid:
                missing.append("TELEON_TWILIO_ACCOUNT_SID")
            if not auth_token:
                missing.append("TELEON_TWILIO_AUTH_TOKEN")
            if not from_number:
                missing.append("TELEON_TWILIO_FROM")
            if missing:
                return ToolResult(
                    success=False,
                    error=(
                        "SMS tool not configured. Missing Twilio credentials. "
                        "Set TELEON_TWILIO_ACCOUNT_SID, TELEON_TWILIO_AUTH_TOKEN, TELEON_TWILIO_FROM."
                    ),
                    tool_name=self.name,
                    metadata={"missing_env": missing},
                )

            if dry_run:
                return ToolResult(
                    success=True,
                    data={"sent": False, "dry_run": True, "to": to, "from": from_number},
                    tool_name=self.name,
                )

            def _send() -> dict:
                client = TwilioClient(account_sid, auth_token)
                msg = client.messages.create(body=message, from_=from_number, to=to)
                return {"sid": msg.sid, "status": msg.status}

            data = await asyncio.to_thread(_send)
            return ToolResult(
                success=True,
                data={"sent": True, "to": to, "from": from_number, **data},
                tool_name=self.name,
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
                    "message": {"type": "string"},
                    "from": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                    "twilio_account_sid": {"type": "string"},
                    "twilio_auth_token": {"type": "string"},
                },
                "required": ["to", "message"]
            },
            returns={"type": "object"},
            tags=["sms", "text", "communication"],
            requires=[
                "Dependency: twilio (install teleon[twilio])",
                "Env: TELEON_TWILIO_ACCOUNT_SID",
                "Env: TELEON_TWILIO_AUTH_TOKEN",
                "Env: TELEON_TWILIO_FROM",
            ],
            examples=[
                {"to": "+10000000000", "message": "Hello", "dry_run": True}
            ],
        )


class SlackMessageTool(BaseTool):
    """Send Slack message."""
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Send Slack message."""
        channel = kwargs.get("channel")
        text = kwargs.get("text")
        dry_run = bool(kwargs.get("dry_run", False))

        token = kwargs.get("token") or os.getenv("TELEON_SLACK_BOT_TOKEN")
        
        try:
            try:
                from slack_sdk import WebClient
                from slack_sdk.errors import SlackApiError
            except ImportError:
                return ToolResult(
                    success=False,
                    error=(
                        "Slack dependency not installed. Install with: pip install teleon[slack] "
                        "(or pip install slack_sdk)."
                    ),
                    tool_name=self.name,
                    metadata={"missing_dependency": "slack_sdk", "install": "teleon[slack]"},
                )

            if not token:
                return ToolResult(
                    success=False,
                    error=(
                        "Slack tool not configured. Missing bot token. "
                        "Set TELEON_SLACK_BOT_TOKEN (xoxb-...)."
                    ),
                    tool_name=self.name,
                    metadata={"missing_env": ["TELEON_SLACK_BOT_TOKEN"]},
                )

            if dry_run:
                return ToolResult(
                    success=True,
                    data={"sent": False, "dry_run": True, "channel": channel},
                    tool_name=self.name,
                )

            def _send() -> dict:
                client = WebClient(token=token)
                resp = client.chat_postMessage(channel=channel, text=text)
                return {
                    "ts": resp.get("ts"),
                    "channel": resp.get("channel"),
                    "message": resp.get("message"),
                }

            try:
                data = await asyncio.to_thread(_send)
            except SlackApiError as e:
                return ToolResult(
                    success=False,
                    error=str(e),
                    tool_name=self.name,
                    metadata={"slack_error": getattr(e, "response", None).data if getattr(e, "response", None) else None},
                )

            return ToolResult(
                success=True,
                data={"sent": True, **data},
                tool_name=self.name,
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
                    "text": {"type": "string"},
                    "token": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                },
                "required": ["channel", "text"]
            },
            returns={"type": "object"},
            tags=["slack", "message", "communication"],
            requires=[
                "Dependency: slack_sdk (install teleon[slack])",
                "Env: TELEON_SLACK_BOT_TOKEN",
            ],
            examples=[
                {"channel": "#general", "text": "Hello", "dry_run": True}
            ],
        )

