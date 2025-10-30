"""
Webhook Signature Validation - Verify webhook authenticity.

Supports validation methods for:
- HMAC SHA256 (Stripe, GitHub, Slack)
- Custom validation schemes
"""

import hmac
import hashlib
from typing import Dict, Optional, Callable

from teleon.core import TeleonException


class SignatureValidationError(TeleonException):
    """Raised when webhook signature validation fails."""
    pass


class WebhookValidator:
    """
    Validates webhook signatures for security.
    
    Ensures webhooks come from legitimate sources by verifying
    cryptographic signatures.
    """
    
    @staticmethod
    def validate_hmac_sha256(
        payload: bytes,
        signature: str,
        secret: str,
        signature_header: str = "sha256="
    ) -> bool:
        """
        Validate HMAC SHA256 signature.
        
        Used by: Stripe, GitHub, Slack
        
        Args:
            payload: Raw request body
            signature: Signature from header
            secret: Webhook secret
            signature_header: Signature format prefix
            
        Returns:
            True if valid
            
        Raises:
            SignatureValidationError: If signature invalid
        """
        # Remove signature header prefix if present
        if signature.startswith(signature_header):
            signature = signature[len(signature_header):]
        
        # Calculate expected signature
        expected = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Constant-time comparison
        if not hmac.compare_digest(expected, signature):
            raise SignatureValidationError("Invalid webhook signature")
        
        return True
    
    @staticmethod
    def validate_github(
        payload: bytes,
        signature: str,
        secret: str
    ) -> bool:
        """
        Validate GitHub webhook signature.
        
        Args:
            payload: Raw request body
            signature: X-Hub-Signature-256 header value
            secret: Webhook secret
            
        Returns:
            True if valid
        """
        return WebhookValidator.validate_hmac_sha256(
            payload, signature, secret, signature_header="sha256="
        )
    
    @staticmethod
    def validate_stripe(
        payload: bytes,
        signature: str,
        secret: str,
        timestamp: Optional[int] = None,
        tolerance: int = 300
    ) -> bool:
        """
        Validate Stripe webhook signature.
        
        Args:
            payload: Raw request body
            signature: Stripe-Signature header value
            secret: Webhook secret
            timestamp: Request timestamp
            tolerance: Timestamp tolerance in seconds
            
        Returns:
            True if valid
        """
        # Parse Stripe signature format: t=timestamp,v1=signature
        parts = {}
        for part in signature.split(","):
            key, value = part.split("=", 1)
            parts[key] = value
        
        sig_timestamp = int(parts.get("t", 0))
        sig_value = parts.get("v1", "")
        
        # Check timestamp tolerance
        if timestamp and abs(timestamp - sig_timestamp) > tolerance:
            raise SignatureValidationError("Webhook timestamp too old")
        
        # Construct signed payload
        signed_payload = f"{sig_timestamp}.{payload.decode()}"
        
        # Validate signature
        return WebhookValidator.validate_hmac_sha256(
            signed_payload.encode(),
            sig_value,
            secret,
            signature_header=""
        )
    
    @staticmethod
    def validate_slack(
        payload: bytes,
        signature: str,
        secret: str,
        timestamp: str
    ) -> bool:
        """
        Validate Slack webhook signature.
        
        Args:
            payload: Raw request body
            signature: X-Slack-Signature header value
            secret: Signing secret
            timestamp: X-Slack-Request-Timestamp header value
            
        Returns:
            True if valid
        """
        # Construct sig_basestring
        sig_basestring = f"v0:{timestamp}:{payload.decode()}"
        
        return WebhookValidator.validate_hmac_sha256(
            sig_basestring.encode(),
            signature,
            secret,
            signature_header="v0="
        )
    
    @staticmethod
    def create_custom_validator(
        validation_func: Callable[[bytes, Dict[str, str]], bool]
    ) -> Callable:
        """
        Create a custom validator function.
        
        Args:
            validation_func: Custom validation function
            
        Returns:
            Validator function
        """
        def validator(payload: bytes, headers: Dict[str, str]) -> bool:
            try:
                return validation_func(payload, headers)
            except Exception as e:
                raise SignatureValidationError(f"Validation failed: {e}") from e
        
        return validator

