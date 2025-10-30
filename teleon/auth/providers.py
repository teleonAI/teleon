"""
OAuth Providers - Pre-configured OAuth2 providers.

Provides ready-to-use configurations for popular services:
- GitHub
- Google
- Slack
- Microsoft
- And more
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class OAuthProvider:
    """OAuth provider configuration."""
    
    name: str
    """Provider name"""
    
    authorization_url: str
    """Authorization endpoint"""
    
    token_url: str
    """Token endpoint"""
    
    userinfo_url: Optional[str] = None
    """User info endpoint"""
    
    default_scope: list[str] = None
    """Default scopes"""
    
    def __post_init__(self):
        if self.default_scope is None:
            self.default_scope = []


# Pre-configured providers
PROVIDERS: Dict[str, OAuthProvider] = {
    "github": OAuthProvider(
        name="GitHub",
        authorization_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        userinfo_url="https://api.github.com/user",
        default_scope=["user:email", "read:user"]
    ),
    
    "google": OAuthProvider(
        name="Google",
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
        default_scope=["openid", "email", "profile"]
    ),
    
    "slack": OAuthProvider(
        name="Slack",
        authorization_url="https://slack.com/oauth/v2/authorize",
        token_url="https://slack.com/api/oauth.v2.access",
        default_scope=["chat:write", "channels:read"]
    ),
    
    "microsoft": OAuthProvider(
        name="Microsoft",
        authorization_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        userinfo_url="https://graph.microsoft.com/v1.0/me",
        default_scope=["User.Read"]
    ),
    
    "spotify": OAuthProvider(
        name="Spotify",
        authorization_url="https://accounts.spotify.com/authorize",
        token_url="https://accounts.spotify.com/api/token",
        userinfo_url="https://api.spotify.com/v1/me",
        default_scope=["user-read-email", "user-read-private"]
    ),
    
    "gitlab": OAuthProvider(
        name="GitLab",
        authorization_url="https://gitlab.com/oauth/authorize",
        token_url="https://gitlab.com/oauth/token",
        userinfo_url="https://gitlab.com/api/v4/user",
        default_scope=["read_user", "api"]
    ),
    
    "linkedin": OAuthProvider(
        name="LinkedIn",
        authorization_url="https://www.linkedin.com/oauth/v2/authorization",
        token_url="https://www.linkedin.com/oauth/v2/accessToken",
        userinfo_url="https://api.linkedin.com/v2/me",
        default_scope=["r_liteprofile", "r_emailaddress"]
    ),
    
    "dropbox": OAuthProvider(
        name="Dropbox",
        authorization_url="https://www.dropbox.com/oauth2/authorize",
        token_url="https://api.dropboxapi.com/oauth2/token",
        default_scope=["files.content.read", "files.content.write"]
    ),
}


def get_provider(name: str) -> Optional[OAuthProvider]:
    """
    Get OAuth provider by name.
    
    Args:
        name: Provider name (e.g., "github", "google")
        
    Returns:
        OAuth provider configuration or None
    """
    return PROVIDERS.get(name.lower())


def list_providers() -> list[str]:
    """
    List available OAuth providers.
    
    Returns:
        List of provider names
    """
    return list(PROVIDERS.keys())


def register_provider(name: str, provider: OAuthProvider):
    """
    Register a custom OAuth provider.
    
    Args:
        name: Provider name
        provider: Provider configuration
    """
    PROVIDERS[name.lower()] = provider

