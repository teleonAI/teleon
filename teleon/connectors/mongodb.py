"""
MongoDB Connector - Connect to MongoDB databases.

Provides:
- Document operations
- Query execution
- Aggregation pipelines
- Collection management
"""

from typing import List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient

from teleon.connectors.base import BaseConnector, ConnectionError


class MongoDBConnector(BaseConnector):
    """
    MongoDB database connector.
    
    Example:
        >>> connector = MongoDBConnector(
        ...     uri="mongodb://localhost:27017",
        ...     database="mydb"
        ... )
        >>> 
        >>> async with connector:
        ...     results = await connector.find("users", {"age": {"$gt": 18}})
    """
    
    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        database: str = "teleon"
    ):
        """
        Initialize MongoDB connector.
        
        Args:
            uri: MongoDB connection URI
            database: Database name
        """
        super().__init__("mongodb")
        
        self.uri = uri
        self.database_name = database
        
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
    
    async def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(self.uri)
            self.database = self.client[self.database_name]
            
            # Test connection
            await self.client.admin.command("ping")
            
            self.connected = True
            self.logger.info(f"Connected to MongoDB: {self.database_name}")
        
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}") from e
    
    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            self.connected = False
            self.logger.info("Disconnected from MongoDB")
    
    async def test_connection(self) -> bool:
        """Test MongoDB connection."""
        try:
            await self.client.admin.command("ping")
            return True
        except:
            return False
    
    async def insert_one(
        self,
        collection: str,
        document: Dict[str, Any]
    ) -> str:
        """
        Insert a single document.
        
        Args:
            collection: Collection name
            document: Document to insert
            
        Returns:
            Inserted document ID
        """
        await self.ensure_connected()
        
        result = await self.database[collection].insert_one(document)
        return str(result.inserted_id)
    
    async def insert_many(
        self,
        collection: str,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Insert multiple documents.
        
        Args:
            collection: Collection name
            documents: List of documents to insert
            
        Returns:
            List of inserted document IDs
        """
        await self.ensure_connected()
        
        result = await self.database[collection].insert_many(documents)
        return [str(id) for id in result.inserted_ids]
    
    async def find(
        self,
        collection: str,
        query: Dict[str, Any],
        limit: Optional[int] = None,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Find documents.
        
        Args:
            collection: Collection name
            query: Query filter
            limit: Maximum number of documents
            skip: Number of documents to skip
            
        Returns:
            List of matching documents
        """
        await self.ensure_connected()
        
        cursor = self.database[collection].find(query).skip(skip)
        
        if limit:
            cursor = cursor.limit(limit)
        
        documents = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string
        for doc in documents:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        
        return documents
    
    async def find_one(
        self,
        collection: str,
        query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find a single document.
        
        Args:
            collection: Collection name
            query: Query filter
            
        Returns:
            Matching document or None
        """
        await self.ensure_connected()
        
        document = await self.database[collection].find_one(query)
        
        if document and "_id" in document:
            document["_id"] = str(document["_id"])
        
        return document
    
    async def update_one(
        self,
        collection: str,
        query: Dict[str, Any],
        update: Dict[str, Any]
    ) -> int:
        """
        Update a single document.
        
        Args:
            collection: Collection name
            query: Query filter
            update: Update operations
            
        Returns:
            Number of documents modified
        """
        await self.ensure_connected()
        
        result = await self.database[collection].update_one(query, {"$set": update})
        return result.modified_count
    
    async def delete_one(
        self,
        collection: str,
        query: Dict[str, Any]
    ) -> int:
        """
        Delete a single document.
        
        Args:
            collection: Collection name
            query: Query filter
            
        Returns:
            Number of documents deleted
        """
        await self.ensure_connected()
        
        result = await self.database[collection].delete_one(query)
        return result.deleted_count
    
    async def aggregate(
        self,
        collection: str,
        pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute aggregation pipeline.
        
        Args:
            collection: Collection name
            pipeline: Aggregation pipeline
            
        Returns:
            Aggregation results
        """
        await self.ensure_connected()
        
        cursor = self.database[collection].aggregate(pipeline)
        results = await cursor.to_list(length=None)
        
        # Convert ObjectId to string
        for result in results:
            if "_id" in result:
                result["_id"] = str(result["_id"])
        
        return results

