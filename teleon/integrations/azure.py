"""Azure Services Integration - Complete Azure ecosystem support."""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio


class AzureStorageClient:
    """
    Azure Blob Storage integration.
    
    Features:
    - Blob upload/download
    - Container management
    - SAS token generation
    - Hierarchical namespace support
    """
    
    def __init__(
        self,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        connection_string: Optional[str] = None,
        use_managed_identity: bool = False
    ):
        """
        Initialize Azure Storage client.
        
        Args:
            account_name: Azure storage account name
            account_key: Azure storage account key
            connection_string: Azure storage connection string
            use_managed_identity: Use Azure Managed Identity
        """
        self.account_name = account_name or os.getenv("AZURE_STORAGE_ACCOUNT")
        self.account_key = account_key or os.getenv("AZURE_STORAGE_KEY")
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.use_managed_identity = use_managed_identity
        
        # Will be initialized lazily
        self._blob_service_client = None
    
    async def _get_client(self):
        """Get Azure Blob Service client."""
        if self._blob_service_client:
            return self._blob_service_client
        
        try:
            from azure.storage.blob.aio import BlobServiceClient
            
            if self.use_managed_identity:
                from azure.identity.aio import DefaultAzureCredential
                credential = DefaultAzureCredential()
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self._blob_service_client = BlobServiceClient(account_url, credential=credential)
            elif self.connection_string:
                self._blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            else:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                from azure.storage.blob import BlobServiceClient
                self._blob_service_client = BlobServiceClient(account_url, credential=self.account_key)
            
            return self._blob_service_client
        except ImportError:
            raise ImportError("azure-storage-blob required. Install: pip install azure-storage-blob")
    
    async def upload_blob(
        self,
        container: str,
        blob_name: str,
        data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Upload data to blob storage."""
        client = await self._get_client()
        blob_client = client.get_blob_client(container=container, blob=blob_name)
        
        await blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings={"content_type": content_type} if content_type else None,
            metadata=metadata
        )
        
        return blob_client.url
    
    async def download_blob(self, container: str, blob_name: str) -> bytes:
        """Download blob data."""
        client = await self._get_client()
        blob_client = client.get_blob_client(container=container, blob=blob_name)
        
        stream = await blob_client.download_blob()
        data = await stream.readall()
        return data
    
    async def list_blobs(self, container: str, prefix: Optional[str] = None) -> List[str]:
        """List blobs in a container."""
        client = await self._get_client()
        container_client = client.get_container_client(container)
        
        blobs = []
        async for blob in container_client.list_blobs(name_starts_with=prefix):
            blobs.append(blob.name)
        
        return blobs
    
    async def delete_blob(self, container: str, blob_name: str):
        """Delete a blob."""
        client = await self._get_client()
        blob_client = client.get_blob_client(container=container, blob=blob_name)
        await blob_client.delete_blob()


class AzureKeyVaultClient:
    """
    Azure Key Vault integration for secrets management.
    
    Features:
    - Secret storage and retrieval
    - Secret rotation
    - Managed identity support
    - Audit logging
    """
    
    def __init__(
        self,
        vault_url: Optional[str] = None,
        use_managed_identity: bool = True
    ):
        """
        Initialize Azure Key Vault client.
        
        Args:
            vault_url: Key Vault URL (e.g., https://your-vault.vault.azure.net/)
            use_managed_identity: Use Azure Managed Identity
        """
        self.vault_url = vault_url or os.getenv("AZURE_KEY_VAULT_URL")
        self.use_managed_identity = use_managed_identity
        
        if not self.vault_url:
            raise ValueError("Azure Key Vault URL is required")
        
        self._client = None
    
    async def _get_client(self):
        """Get Azure Key Vault client."""
        if self._client:
            return self._client
        
        try:
            from azure.keyvault.secrets.aio import SecretClient
            from azure.identity.aio import DefaultAzureCredential
            
            credential = DefaultAzureCredential()
            self._client = SecretClient(vault_url=self.vault_url, credential=credential)
            return self._client
        except ImportError:
            raise ImportError("azure-keyvault-secrets required. Install: pip install azure-keyvault-secrets azure-identity")
    
    async def get_secret(self, name: str) -> str:
        """Get a secret from Key Vault."""
        client = await self._get_client()
        secret = await client.get_secret(name)
        return secret.value
    
    async def set_secret(self, name: str, value: str) -> None:
        """Set a secret in Key Vault."""
        client = await self._get_client()
        await client.set_secret(name, value)
    
    async def delete_secret(self, name: str) -> None:
        """Delete a secret from Key Vault."""
        client = await self._get_client()
        await client.begin_delete_secret(name)


class AzureCosmosDBClient:
    """
    Azure Cosmos DB integration.
    
    Features:
    - Document operations
    - Query support
    - Partition key handling
    - Throughput management
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        database_name: str = "teleon",
        container_name: str = "agents",
        use_managed_identity: bool = False
    ):
        """
        Initialize Azure Cosmos DB client.
        
        Args:
            endpoint: Cosmos DB endpoint
            key: Cosmos DB key
            database_name: Database name
            container_name: Container name
            use_managed_identity: Use Azure Managed Identity
        """
        self.endpoint = endpoint or os.getenv("AZURE_COSMOS_ENDPOINT")
        self.key = key or os.getenv("AZURE_COSMOS_KEY")
        self.database_name = database_name
        self.container_name = container_name
        self.use_managed_identity = use_managed_identity
        
        if not self.endpoint:
            raise ValueError("Azure Cosmos DB endpoint is required")
        
        self._client = None
        self._database = None
        self._container = None
    
    async def _get_client(self):
        """Get Azure Cosmos DB client."""
        if self._client:
            return self._client
        
        try:
            from azure.cosmos.aio import CosmosClient
            
            if self.use_managed_identity:
                from azure.identity.aio import DefaultAzureCredential
                credential = DefaultAzureCredential()
                self._client = CosmosClient(self.endpoint, credential=credential)
            else:
                self._client = CosmosClient(self.endpoint, credential=self.key)
            
            self._database = self._client.get_database_client(self.database_name)
            self._container = self._database.get_container_client(self.container_name)
            
            return self._client
        except ImportError:
            raise ImportError("azure-cosmos required. Install: pip install azure-cosmos")
    
    async def create_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Create an item in Cosmos DB."""
        await self._get_client()
        return await self._container.create_item(body=item)
    
    async def read_item(self, item_id: str, partition_key: str) -> Dict[str, Any]:
        """Read an item from Cosmos DB."""
        await self._get_client()
        return await self._container.read_item(item=item_id, partition_key=partition_key)
    
    async def query_items(self, query: str, parameters: Optional[List] = None) -> List[Dict[str, Any]]:
        """Query items from Cosmos DB."""
        await self._get_client()
        items = []
        
        async for item in self._container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ):
            items.append(item)
        
        return items
    
    async def upsert_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert an item in Cosmos DB."""
        await self._get_client()
        return await self._container.upsert_item(body=item)
    
    async def delete_item(self, item_id: str, partition_key: str) -> None:
        """Delete an item from Cosmos DB."""
        await self._get_client()
        await self._container.delete_item(item=item_id, partition_key=partition_key)


class AzureServiceBusClient:
    """
    Azure Service Bus integration for message queuing.
    
    Features:
    - Queue and topic support
    - Message sending and receiving
    - Session support
    - Dead letter handling
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        queue_name: Optional[str] = None,
        use_managed_identity: bool = False
    ):
        """
        Initialize Azure Service Bus client.
        
        Args:
            connection_string: Service Bus connection string
            queue_name: Default queue name
            use_managed_identity: Use Azure Managed Identity
        """
        self.connection_string = connection_string or os.getenv("AZURE_SERVICE_BUS_CONNECTION_STRING")
        self.queue_name = queue_name
        self.use_managed_identity = use_managed_identity
        
        self._client = None
    
    async def _get_client(self):
        """Get Azure Service Bus client."""
        if self._client:
            return self._client
        
        try:
            from azure.servicebus.aio import ServiceBusClient
            
            if self.use_managed_identity:
                from azure.identity.aio import DefaultAzureCredential
                credential = DefaultAzureCredential()
                namespace = os.getenv("AZURE_SERVICE_BUS_NAMESPACE")
                self._client = ServiceBusClient(
                    fully_qualified_namespace=f"{namespace}.servicebus.windows.net",
                    credential=credential
                )
            else:
                self._client = ServiceBusClient.from_connection_string(self.connection_string)
            
            return self._client
        except ImportError:
            raise ImportError("azure-servicebus required. Install: pip install azure-servicebus")
    
    async def send_message(self, queue_name: str, message: str, properties: Optional[Dict] = None):
        """Send a message to a queue."""
        client = await self._get_client()
        
        async with client.get_queue_sender(queue_name) as sender:
            from azure.servicebus import ServiceBusMessage
            msg = ServiceBusMessage(message)
            
            if properties:
                msg.application_properties = properties
            
            await sender.send_messages(msg)
    
    async def receive_messages(
        self,
        queue_name: str,
        max_messages: int = 10,
        max_wait_time: int = 5
    ) -> List[str]:
        """Receive messages from a queue."""
        client = await self._get_client()
        messages = []
        
        async with client.get_queue_receiver(queue_name) as receiver:
            received_msgs = await receiver.receive_messages(
                max_message_count=max_messages,
                max_wait_time=max_wait_time
            )
            
            for msg in received_msgs:
                messages.append(str(msg))
                await receiver.complete_message(msg)
        
        return messages


class AzureMonitorClient:
    """
    Azure Monitor integration for logging and metrics.
    
    Features:
    - Application Insights telemetry
    - Custom metrics
    - Log Analytics
    - Distributed tracing
    """
    
    def __init__(
        self,
        instrumentation_key: Optional[str] = None,
        connection_string: Optional[str] = None
    ):
        """
        Initialize Azure Monitor client.
        
        Args:
            instrumentation_key: Application Insights instrumentation key
            connection_string: Application Insights connection string
        """
        self.instrumentation_key = instrumentation_key or os.getenv("APPINSIGHTS_INSTRUMENTATION_KEY")
        self.connection_string = connection_string or os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        
        self._telemetry_client = None
    
    def _get_client(self):
        """Get Azure Monitor telemetry client."""
        if self._telemetry_client:
            return self._telemetry_client
        
        try:
            from applicationinsights import TelemetryClient
            
            if self.connection_string:
                # Parse connection string for instrumentation key
                parts = dict(item.split("=") for item in self.connection_string.split(";"))
                key = parts.get("InstrumentationKey", self.instrumentation_key)
            else:
                key = self.instrumentation_key
            
            self._telemetry_client = TelemetryClient(key)
            return self._telemetry_client
        except ImportError:
            raise ImportError("applicationinsights required. Install: pip install applicationinsights")
    
    def track_event(self, name: str, properties: Optional[Dict] = None, measurements: Optional[Dict] = None):
        """Track a custom event."""
        client = self._get_client()
        client.track_event(name, properties, measurements)
        client.flush()
    
    def track_metric(self, name: str, value: float, properties: Optional[Dict] = None):
        """Track a custom metric."""
        client = self._get_client()
        client.track_metric(name, value, properties=properties)
        client.flush()
    
    def track_trace(self, message: str, severity: str = "INFO", properties: Optional[Dict] = None):
        """Track a trace message."""
        client = self._get_client()
        client.track_trace(message, severity=severity, properties=properties)
        client.flush()
    
    def track_exception(self, exception: Exception, properties: Optional[Dict] = None):
        """Track an exception."""
        client = self._get_client()
        client.track_exception(type(exception), exception, None, properties=properties)
        client.flush()


# Convenience function to get all Azure clients
def get_azure_clients(
    use_managed_identity: bool = True
) -> Dict[str, Any]:
    """
    Get all configured Azure service clients.
    
    Args:
        use_managed_identity: Use Azure Managed Identity for authentication
    
    Returns:
        Dictionary of Azure service clients
    """
    clients = {}
    
    # Storage
    if os.getenv("AZURE_STORAGE_ACCOUNT") or os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
        clients["storage"] = AzureStorageClient(use_managed_identity=use_managed_identity)
    
    # Key Vault
    if os.getenv("AZURE_KEY_VAULT_URL"):
        clients["keyvault"] = AzureKeyVaultClient(use_managed_identity=use_managed_identity)
    
    # Cosmos DB
    if os.getenv("AZURE_COSMOS_ENDPOINT"):
        clients["cosmos"] = AzureCosmosDBClient(use_managed_identity=use_managed_identity)
    
    # Service Bus
    if os.getenv("AZURE_SERVICE_BUS_CONNECTION_STRING") or os.getenv("AZURE_SERVICE_BUS_NAMESPACE"):
        clients["servicebus"] = AzureServiceBusClient(use_managed_identity=use_managed_identity)
    
    # Monitor
    if os.getenv("APPINSIGHTS_INSTRUMENTATION_KEY") or os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        clients["monitor"] = AzureMonitorClient()
    
    return clients

