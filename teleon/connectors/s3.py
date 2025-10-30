"""
S3 Connector - Connect to AWS S3 storage.

Provides:
- File upload/download
- Bucket operations
- Presigned URLs
- Batch operations
"""

from typing import List, Dict, Any, Optional
import aioboto3

from teleon.connectors.base import BaseConnector, ConnectionError


class S3Connector(BaseConnector):
    """
    AWS S3 storage connector.
    
    Example:
        >>> connector = S3Connector(
        ...     aws_access_key_id="your-key",
        ...     aws_secret_access_key="your-secret",
        ...     region_name="us-east-1"
        ... )
        >>> 
        >>> async with connector:
        ...     await connector.upload_file("bucket", "key", "path/to/file")
    """
    
    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str = "us-east-1"
    ):
        """
        Initialize S3 connector.
        
        Args:
            aws_access_key_id: AWS access key
            aws_secret_access_key: AWS secret key
            region_name: AWS region
        """
        super().__init__("s3")
        
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        
        self.session = None
        self.client = None
    
    async def connect(self):
        """Connect to S3."""
        try:
            self.session = aioboto3.Session()
            self.client = await self.session.client(
                "s3",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name
            ).__aenter__()
            
            self.connected = True
            self.logger.info(f"Connected to S3: {self.region_name}")
        
        except Exception as e:
            raise ConnectionError(f"Failed to connect to S3: {e}") from e
    
    async def disconnect(self):
        """Disconnect from S3."""
        if self.client:
            await self.client.__aexit__(None, None, None)
            self.connected = False
            self.logger.info("Disconnected from S3")
    
    async def test_connection(self) -> bool:
        """Test S3 connection."""
        try:
            await self.client.list_buckets()
            return True
        except:
            return False
    
    async def upload_file(
        self,
        bucket: str,
        key: str,
        file_path: str,
        metadata: Optional[Dict[str, str]] = None
    ):
        """
        Upload a file to S3.
        
        Args:
            bucket: Bucket name
            key: Object key
            file_path: Local file path
            metadata: Optional metadata
        """
        await self.ensure_connected()
        
        extra_args = {}
        if metadata:
            extra_args["Metadata"] = metadata
        
        await self.client.upload_file(file_path, bucket, key, ExtraArgs=extra_args)
        self.logger.info(f"Uploaded file to s3://{bucket}/{key}")
    
    async def download_file(
        self,
        bucket: str,
        key: str,
        file_path: str
    ):
        """
        Download a file from S3.
        
        Args:
            bucket: Bucket name
            key: Object key
            file_path: Local file path
        """
        await self.ensure_connected()
        
        await self.client.download_file(bucket, key, file_path)
        self.logger.info(f"Downloaded file from s3://{bucket}/{key}")
    
    async def upload_fileobj(
        self,
        bucket: str,
        key: str,
        file_obj: Any,
        metadata: Optional[Dict[str, str]] = None
    ):
        """
        Upload a file-like object to S3.
        
        Args:
            bucket: Bucket name
            key: Object key
            file_obj: File-like object
            metadata: Optional metadata
        """
        await self.ensure_connected()
        
        extra_args = {}
        if metadata:
            extra_args["Metadata"] = metadata
        
        await self.client.upload_fileobj(file_obj, bucket, key, ExtraArgs=extra_args)
        self.logger.info(f"Uploaded object to s3://{bucket}/{key}")
    
    async def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List objects in bucket.
        
        Args:
            bucket: Bucket name
            prefix: Key prefix filter
            max_keys: Maximum number of keys
            
        Returns:
            List of objects
        """
        await self.ensure_connected()
        
        response = await self.client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=max_keys
        )
        
        return response.get("Contents", [])
    
    async def delete_object(
        self,
        bucket: str,
        key: str
    ):
        """
        Delete an object.
        
        Args:
            bucket: Bucket name
            key: Object key
        """
        await self.ensure_connected()
        
        await self.client.delete_object(Bucket=bucket, Key=key)
        self.logger.info(f"Deleted object s3://{bucket}/{key}")
    
    async def generate_presigned_url(
        self,
        bucket: str,
        key: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate presigned URL.
        
        Args:
            bucket: Bucket name
            key: Object key
            expiration: URL expiration in seconds
            
        Returns:
            Presigned URL
        """
        await self.ensure_connected()
        
        url = await self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiration
        )
        
        return url

