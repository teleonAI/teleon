"""
PostgreSQL Connector - Connect to PostgreSQL databases.

Provides:
- Query execution
- Transaction management
- Connection pooling
- Batch operations
"""

from typing import List, Dict, Any, Optional
import asyncpg

from teleon.connectors.base import BaseConnector, ConnectionError


class PostgreSQLConnector(BaseConnector):
    """
    PostgreSQL database connector.
    
    Example:
        >>> connector = PostgreSQLConnector(
        ...     host="localhost",
        ...     port=5432,
        ...     database="mydb",
        ...     user="user",
        ...     password="password"
        ... )
        >>> 
        >>> async with connector:
        ...     results = await connector.query("SELECT * FROM users")
    """
    
    def __init__(
        self,
        host: str,
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres",
        password: str = "",
        min_pool_size: int = 1,
        max_pool_size: int = 10
    ):
        """
        Initialize PostgreSQL connector.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
            min_pool_size: Minimum pool size
            max_pool_size: Maximum pool size
        """
        super().__init__("postgresql")
        
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Connect to PostgreSQL database."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size
            )
            
            self.connected = True
            self.logger.info(f"Connected to PostgreSQL: {self.host}:{self.port}/{self.database}")
        
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e
    
    async def disconnect(self):
        """Disconnect from PostgreSQL database."""
        if self.pool:
            await self.pool.close()
            self.connected = False
            self.logger.info("Disconnected from PostgreSQL")
    
    async def test_connection(self) -> bool:
        """Test PostgreSQL connection."""
        try:
            await self.query("SELECT 1")
            return True
        except:
            return False
    
    async def query(
        self,
        sql: str,
        *args
    ) -> List[Dict[str, Any]]:
        """
        Execute a query.
        
        Args:
            sql: SQL query
            *args: Query parameters
            
        Returns:
            List of result rows as dictionaries
        """
        await self.ensure_connected()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
            return [dict(row) for row in rows]
    
    async def execute(
        self,
        sql: str,
        *args
    ) -> str:
        """
        Execute a command (INSERT, UPDATE, DELETE).
        
        Args:
            sql: SQL command
            *args: Command parameters
            
        Returns:
            Status message
        """
        await self.ensure_connected()
        
        async with self.pool.acquire() as conn:
            result = await conn.execute(sql, *args)
            self.logger.info(f"Executed: {result}")
            return result
    
    async def insert(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> int:
        """
        Insert a row.
        
        Args:
            table: Table name
            data: Data to insert
            
        Returns:
            Inserted row ID
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f"${i+1}" for i in range(len(data)))
        
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING id"
        
        await self.ensure_connected()
        
        async with self.pool.acquire() as conn:
            row_id = await conn.fetchval(sql, *data.values())
            return row_id
    
    async def batch_insert(
        self,
        table: str,
        records: List[Dict[str, Any]]
    ) -> int:
        """
        Insert multiple rows.
        
        Args:
            table: Table name
            records: List of records to insert
            
        Returns:
            Number of rows inserted
        """
        if not records:
            return 0
        
        columns = list(records[0].keys())
        column_names = ", ".join(columns)
        
        await self.ensure_connected()
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                count = 0
                for record in records:
                    placeholders = ", ".join(f"${i+1}" for i in range(len(columns)))
                    sql = f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})"
                    await conn.execute(sql, *[record[col] for col in columns])
                    count += 1
                
                return count

