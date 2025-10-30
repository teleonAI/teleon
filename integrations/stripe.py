"""
Stripe Integration - Payment processing and webhook handling.

Provides:
- Create customers and subscriptions
- Process payments
- Handle webhooks
- Manage products and prices
- Retrieve payment information
"""

from typing import Dict, Any, List, Optional
import httpx

from teleon.integrations.base import (
    BaseIntegration,
    IntegrationConfig,
    IntegrationError,
    AuthenticationError,
)


class StripeIntegration(BaseIntegration):
    """
    Stripe integration for payment processing.
    
    Example:
        >>> config = IntegrationConfig(
        ...     name="stripe",
        ...     api_key="sk_test_..."
        ... )
        >>> stripe = StripeIntegration(config)
        >>> customer = await stripe.create_customer("user@example.com")
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize Stripe integration."""
        if not config.base_url:
            config.base_url = "https://api.stripe.com/v1"
        super().__init__(config)
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            auth=(config.api_key, "")
        )
    
    async def authenticate(self) -> bool:
        """Authenticate with Stripe API."""
        try:
            response = await self.client.get("/balance")
            
            if response.status_code != 200:
                raise AuthenticationError(f"Stripe auth failed: {response.text}")
            
            self._authenticated = True
            self.logger.info("Stripe authentication successful")
            return True
        
        except Exception as e:
            raise AuthenticationError(f"Stripe authentication failed: {e}") from e
    
    async def test_connection(self) -> bool:
        """Test connection to Stripe."""
        return await self.authenticate()
    
    async def create_customer(
        self,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new customer.
        
        Args:
            email: Customer email
            name: Customer name
            metadata: Additional metadata
            
        Returns:
            Created customer information
        """
        await self.ensure_authenticated()
        
        data = {"email": email}
        
        if name:
            data["name"] = name
        if metadata:
            for key, value in metadata.items():
                data[f"metadata[{key}]"] = value
        
        async def _create():
            response = await self.client.post("/customers", data=data)
            
            if response.status_code not in [200, 201]:
                raise IntegrationError(f"Failed to create customer: {response.text}")
            
            return response.json()
        
        return await self.execute_with_retry(_create)
    
    async def create_payment_intent(
        self,
        amount: int,
        currency: str = "usd",
        customer: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a payment intent.
        
        Args:
            amount: Amount in cents
            currency: Currency code (e.g., "usd")
            customer: Customer ID
            metadata: Additional metadata
            
        Returns:
            Payment intent information
        """
        await self.ensure_authenticated()
        
        data = {
            "amount": amount,
            "currency": currency
        }
        
        if customer:
            data["customer"] = customer
        if metadata:
            for key, value in metadata.items():
                data[f"metadata[{key}]"] = value
        
        async def _create():
            response = await self.client.post("/payment_intents", data=data)
            
            if response.status_code not in [200, 201]:
                raise IntegrationError(f"Failed to create payment intent: {response.text}")
            
            return response.json()
        
        return await self.execute_with_retry(_create)
    
    async def create_subscription(
        self,
        customer: str,
        price: str,
        trial_period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a subscription.
        
        Args:
            customer: Customer ID
            price: Price ID
            trial_period_days: Trial period in days
            
        Returns:
            Subscription information
        """
        await self.ensure_authenticated()
        
        data = {
            "customer": customer,
            "items[0][price]": price
        }
        
        if trial_period_days:
            data["trial_period_days"] = trial_period_days
        
        async def _create():
            response = await self.client.post("/subscriptions", data=data)
            
            if response.status_code not in [200, 201]:
                raise IntegrationError(f"Failed to create subscription: {response.text}")
            
            return response.json()
        
        return await self.execute_with_retry(_create)
    
    async def retrieve_charge(self, charge_id: str) -> Dict[str, Any]:
        """
        Retrieve a charge.
        
        Args:
            charge_id: Charge ID
            
        Returns:
            Charge information
        """
        await self.ensure_authenticated()
        
        async def _retrieve():
            response = await self.client.get(f"/charges/{charge_id}")
            
            if response.status_code != 200:
                raise IntegrationError(f"Failed to retrieve charge: {response.text}")
            
            return response.json()
        
        return await self.execute_with_retry(_retrieve)
    
    async def list_customers(
        self,
        limit: int = 10,
        starting_after: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List customers.
        
        Args:
            limit: Number of customers to return
            starting_after: Cursor for pagination
            
        Returns:
            List of customers
        """
        await self.ensure_authenticated()
        
        params = {"limit": limit}
        if starting_after:
            params["starting_after"] = starting_after
        
        async def _list():
            response = await self.client.get("/customers", params=params)
            
            if response.status_code != 200:
                raise IntegrationError(f"Failed to list customers: {response.text}")
            
            data = response.json()
            return data.get("data", [])
        
        return await self.execute_with_retry(_list)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

