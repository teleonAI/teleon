"""
Teleon Authentication - OAuth2 and authentication management.

Provides:
- OAuth2 client for third-party services
- OAuth provider configurations
- Secure token storage and refresh
"""

from teleon.auth.oauth import OAuth2Client, OAuth2Token
from teleon.auth.providers import OAuthProvider, get_provider
from teleon.auth.token_store import TokenStore, get_token_store


__all__ = [
    "OAuth2Client",
    "OAuth2Token",
    "OAuthProvider",
    "get_provider",
    "TokenStore",
    "get_token_store",
]

