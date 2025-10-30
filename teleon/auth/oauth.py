"""
OAuth2 Client - Complete OAuth2 flow implementation.

Provides:
- Authorization URL generation
- Token exchange
- Token refresh
- Scope management
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import httpx
from urllib.parse import urlencode

from teleon.core import StructuredLogger, LogLevel, TeleonException


class OAuth2Error(TeleonException):
    """Raised when OAuth2 operation fails."""
    pass


@dataclass
class OAuth2Token:
    """OAuth2 token information."""
    
    access_token: str
    """Access token"""
    
    token_type: str = "Bearer"
    """Token type"""
    
    expires_in: Optional[int] = None
    """Token expiration in seconds"""
    
    refresh_token: Optional[str] = None
    """Refresh token"""
    
    scope: Optional[str] = None
    """Granted scopes"""
    
    created_at: datetime = None
    """Token creation time"""
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_in is None:
            return False
        
        expiry_time = self.created_at + timedelta(seconds=self.expires_in)
        return datetime.utcnow() >= expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OAuth2Token":
        """Create from dictionary."""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class OAuth2Client:
    """
    OAuth2 client for third-party service authentication.
    
    Implements complete OAuth2 authorization code flow.
    
    Example:
        >>> client = OAuth2Client(
        ...     client_id="your-client-id",
        ...     client_secret="your-client-secret",
        ...     authorization_url="https://provider.com/oauth/authorize",
        ...     token_url="https://provider.com/oauth/token",
        ...     redirect_uri="https://your-app.com/callback"
        ... )
        >>> 
        >>> # Step 1: Get authorization URL
        >>> auth_url = client.get_authorization_url(scope=["read", "write"])
        >>> 
        >>> # Step 2: Exchange code for token (after user authorizes)
        >>> token = await client.exchange_code("authorization_code")
        >>> 
        >>> # Step 3: Use token
        >>> response = await client.request("GET", "https://api.provider.com/user")
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        authorization_url: str,
        token_url: str,
        redirect_uri: str,
        scope: Optional[List[str]] = None
    ):
        """
        Initialize OAuth2 client.
        
        Args:
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            authorization_url: Authorization endpoint URL
            token_url: Token endpoint URL
            redirect_uri: Redirect URI
            scope: Default scopes
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorization_url = authorization_url
        self.token_url = token_url
        self.redirect_uri = redirect_uri
        self.scope = scope or []
        
        self.logger = StructuredLogger("oauth2_client", LogLevel.INFO)
        self.http_client = httpx.AsyncClient(timeout=30)
    
    def get_authorization_url(
        self,
        state: Optional[str] = None,
        scope: Optional[List[str]] = None,
        **extra_params
    ) -> str:
        """
        Generate authorization URL for user consent.
        
        Args:
            state: CSRF protection state
            scope: OAuth scopes to request
            **extra_params: Additional query parameters
            
        Returns:
            Authorization URL
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(scope or self.scope),
        }
        
        if state:
            params["state"] = state
        
        params.update(extra_params)
        
        url = f"{self.authorization_url}?{urlencode(params)}"
        
        self.logger.info("Generated authorization URL")
        
        return url
    
    async def exchange_code(
        self,
        code: str,
        **extra_params
    ) -> OAuth2Token:
        """
        Exchange authorization code for access token.
        
        Args:
            code: Authorization code
            **extra_params: Additional parameters
            
        Returns:
            OAuth2 token
            
        Raises:
            OAuth2Error: If token exchange fails
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        
        data.update(extra_params)
        
        try:
            response = await self.http_client.post(self.token_url, data=data)
            
            if response.status_code != 200:
                raise OAuth2Error(
                    f"Token exchange failed: {response.status_code} - {response.text}"
                )
            
            token_data = response.json()
            token = OAuth2Token(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in"),
                refresh_token=token_data.get("refresh_token"),
                scope=token_data.get("scope")
            )
            
            self.logger.info("Successfully exchanged code for token")
            
            return token
        
        except Exception as e:
            raise OAuth2Error(f"Token exchange failed: {e}") from e
    
    async def refresh_token(
        self,
        refresh_token: str
    ) -> OAuth2Token:
        """
        Refresh an access token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New OAuth2 token
            
        Raises:
            OAuth2Error: If refresh fails
        """
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        
        try:
            response = await self.http_client.post(self.token_url, data=data)
            
            if response.status_code != 200:
                raise OAuth2Error(
                    f"Token refresh failed: {response.status_code} - {response.text}"
                )
            
            token_data = response.json()
            token = OAuth2Token(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in"),
                refresh_token=token_data.get("refresh_token", refresh_token),
                scope=token_data.get("scope")
            )
            
            self.logger.info("Successfully refreshed token")
            
            return token
        
        except Exception as e:
            raise OAuth2Error(f"Token refresh failed: {e}") from e
    
    async def request(
        self,
        method: str,
        url: str,
        token: OAuth2Token,
        **kwargs
    ) -> httpx.Response:
        """
        Make authenticated HTTP request.
        
        Args:
            method: HTTP method
            url: Request URL
            token: OAuth2 token
            **kwargs: Additional request parameters
            
        Returns:
            HTTP response
        """
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"{token.token_type} {token.access_token}"
        
        response = await self.http_client.request(
            method, url, headers=headers, **kwargs
        )
        
        return response
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()

