"""
Slack Integration - Send messages, manage channels, handle webhooks.

Provides:
- Send messages to channels
- Create and manage channels
- Handle incoming webhooks
- File uploads
- User management
"""

from typing import Dict, Any, List, Optional
import json
import httpx

from teleon.integrations.base import (
    BaseIntegration,
    IntegrationConfig,
    IntegrationError,
    AuthenticationError,
)


class SlackIntegration(BaseIntegration):
    """
    Slack integration for messaging and automation.
    
    Example:
        >>> config = IntegrationConfig(
        ...     name="slack",
        ...     api_key="xoxb-your-bot-token"
        ... )
        >>> slack = SlackIntegration(config)
        >>> await slack.send_message("#general", "Hello from Teleon!")
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize Slack integration."""
        if not config.base_url:
            config.base_url = "https://slack.com/api"
        super().__init__(config)
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={"Authorization": f"Bearer {config.api_key}"}
        )
    
    async def authenticate(self) -> bool:
        """Authenticate with Slack API."""
        try:
            response = await self.client.post("/auth.test")
            data = response.json()
            
            if not data.get("ok"):
                raise AuthenticationError(f"Slack auth failed: {data.get('error')}")
            
            self._authenticated = True
            self.logger.info("Slack authentication successful")
            return True
        
        except Exception as e:
            raise AuthenticationError(f"Slack authentication failed: {e}") from e
    
    async def test_connection(self) -> bool:
        """Test connection to Slack."""
        return await self.authenticate()
    
    async def send_message(
        self,
        channel: str,
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
        thread_ts: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message to a Slack channel.
        
        Args:
            channel: Channel ID or name (e.g., "#general")
            text: Message text
            blocks: Optional message blocks for rich formatting
            thread_ts: Optional thread timestamp to reply to
            
        Returns:
            Response from Slack API
            
        Raises:
            IntegrationError: If send fails
        """
        await self.ensure_authenticated()
        
        payload = {
            "channel": channel,
            "text": text,
        }
        
        if blocks:
            payload["blocks"] = blocks
        
        if thread_ts:
            payload["thread_ts"] = thread_ts
        
        async def _send():
            response = await self.client.post("/chat.postMessage", json=payload)
            data = response.json()
            
            if not data.get("ok"):
                raise IntegrationError(f"Failed to send message: {data.get('error')}")
            
            return data
        
        return await self.execute_with_retry(_send)
    
    async def create_channel(
        self,
        name: str,
        is_private: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new Slack channel.
        
        Args:
            name: Channel name (without #)
            is_private: Whether the channel should be private
            
        Returns:
            Channel information
        """
        await self.ensure_authenticated()
        
        endpoint = "/conversations.create"
        payload = {
            "name": name,
            "is_private": is_private
        }
        
        async def _create():
            response = await self.client.post(endpoint, json=payload)
            data = response.json()
            
            if not data.get("ok"):
                raise IntegrationError(f"Failed to create channel: {data.get('error')}")
            
            return data["channel"]
        
        return await self.execute_with_retry(_create)
    
    async def upload_file(
        self,
        channels: List[str],
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        filename: Optional[str] = None,
        title: Optional[str] = None,
        initial_comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a file to Slack.
        
        Args:
            channels: List of channel IDs/names
            file_path: Path to file (if uploading from file)
            content: File content as string (if creating file)
            filename: Filename
            title: File title
            initial_comment: Initial comment
            
        Returns:
            File information
        """
        await self.ensure_authenticated()
        
        payload = {
            "channels": ",".join(channels)
        }
        
        if content:
            payload["content"] = content
        if filename:
            payload["filename"] = filename
        if title:
            payload["title"] = title
        if initial_comment:
            payload["initial_comment"] = initial_comment
        
        async def _upload():
            if file_path:
                with open(file_path, "rb") as f:
                    files = {"file": f}
                    response = await self.client.post(
                        "/files.upload",
                        data=payload,
                        files=files
                    )
            else:
                response = await self.client.post("/files.upload", data=payload)
            
            data = response.json()
            
            if not data.get("ok"):
                raise IntegrationError(f"Failed to upload file: {data.get('error')}")
            
            return data["file"]
        
        return await self.execute_with_retry(_upload)
    
    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """
        Get information about a user.
        
        Args:
            user_id: User ID
            
        Returns:
            User information
        """
        await self.ensure_authenticated()
        
        async def _get():
            response = await self.client.get(f"/users.info?user={user_id}")
            data = response.json()
            
            if not data.get("ok"):
                raise IntegrationError(f"Failed to get user info: {data.get('error')}")
            
            return data["user"]
        
        return await self.execute_with_retry(_get)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

