"""
GCS Connector - Connect to Google Cloud Storage.

Provides:
- File upload/download
- Bucket operations
- Signed URLs
- Batch operations
"""

from typing import List, Dict, Any, Optional
from google.cloud import storage
from google.oauth2 import service_account

from teleon.connectors.base import BaseConnector, ConnectionError


class GCSConnector(BaseConnector):
    """
    Google Cloud Storage connector.
    
    Example:
        >>> connector = GCSConnector(
        ...     project_id="my-project",
        ...     credentials_path="path/to/credentials.json"
        ... )
        >>> 
        >>> async with connector:
        ...     await connector.upload_file("bucket", "key", "path/to/file")
    """
    
    def __init__(
        self,
        project_id: str,
        credentials_path: Optional[str] = None,
        credentials_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize GCS connector.
        
        Args:
            project_id: GCP project ID
            credentials_path: Path to credentials JSON file
            credentials_dict: Credentials as dictionary
        """
        super().__init__("gcs")
        
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.credentials_dict = credentials_dict
        
        self.client: Optional[storage.Client] = None
    
    async def connect(self):
        """Connect to GCS."""
        try:
            if self.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self.client = storage.Client(
                    project=self.project_id,
                    credentials=credentials
                )
            elif self.credentials_dict:
                credentials = service_account.Credentials.from_service_account_info(
                    self.credentials_dict
                )
                self.client = storage.Client(
                    project=self.project_id,
                    credentials=credentials
                )
            else:
                # Use default credentials
                self.client = storage.Client(project=self.project_id)
            
            self.connected = True
            self.logger.info(f"Connected to GCS: {self.project_id}")
        
        except Exception as e:
            raise ConnectionError(f"Failed to connect to GCS: {e}") from e
    
    async def disconnect(self):
        """Disconnect from GCS."""
        if self.client:
            self.client.close()
            self.connected = False
            self.logger.info("Disconnected from GCS")
    
    async def test_connection(self) -> bool:
        """Test GCS connection."""
        try:
            list(self.client.list_buckets(max_results=1))
            return True
        except:
            return False
    
    async def upload_file(
        self,
        bucket_name: str,
        blob_name: str,
        file_path: str,
        metadata: Optional[Dict[str, str]] = None
    ):
        """
        Upload a file to GCS.
        
        Args:
            bucket_name: Bucket name
            blob_name: Blob name
            file_path: Local file path
            metadata: Optional metadata
        """
        await self.ensure_connected()
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        if metadata:
            blob.metadata = metadata
        
        blob.upload_from_filename(file_path)
        self.logger.info(f"Uploaded file to gs://{bucket_name}/{blob_name}")
    
    async def download_file(
        self,
        bucket_name: str,
        blob_name: str,
        file_path: str
    ):
        """
        Download a file from GCS.
        
        Args:
            bucket_name: Bucket name
            blob_name: Blob name
            file_path: Local file path
        """
        await self.ensure_connected()
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.download_to_filename(file_path)
        self.logger.info(f"Downloaded file from gs://{bucket_name}/{blob_name}")
    
    async def list_blobs(
        self,
        bucket_name: str,
        prefix: str = "",
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List blobs in bucket.
        
        Args:
            bucket_name: Bucket name
            prefix: Blob name prefix filter
            max_results: Maximum number of results
            
        Returns:
            List of blobs
        """
        await self.ensure_connected()
        
        bucket = self.client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)
        
        return [
            {
                "name": blob.name,
                "size": blob.size,
                "updated": blob.updated,
                "content_type": blob.content_type
            }
            for blob in blobs
        ]
    
    async def delete_blob(
        self,
        bucket_name: str,
        blob_name: str
    ):
        """
        Delete a blob.
        
        Args:
            bucket_name: Bucket name
            blob_name: Blob name
        """
        await self.ensure_connected()
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.delete()
        self.logger.info(f"Deleted blob gs://{bucket_name}/{blob_name}")
    
    async def generate_signed_url(
        self,
        bucket_name: str,
        blob_name: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate signed URL.
        
        Args:
            bucket_name: Bucket name
            blob_name: Blob name
            expiration: URL expiration in seconds
            
        Returns:
            Signed URL
        """
        await self.ensure_connected()
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        url = blob.generate_signed_url(
            version="v4",
            expiration=expiration,
            method="GET"
        )
        
        return url

