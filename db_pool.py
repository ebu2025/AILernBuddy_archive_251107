"""SQLite connection pool for better resource management."""
import logging
import sqlite3
import threading
from contextlib import contextmanager
from queue import Empty, Queue
from typing import Generator

logger = logging.getLogger(__name__)

class SQLiteConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, database: str, max_connections: int = 5):
        self.database = database
        self.max_connections = max_connections
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._created_connections = 0
        
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with proper settings."""
        conn = sqlite3.connect(self.database)
        conn.row_factory = sqlite3.Row
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
        
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a connection from the pool or create a new one if needed."""
        connection = None
        try:
            connection = self._pool.get(block=False)
        except Empty:
            with self._lock:
                if self._created_connections < self.max_connections:
                    connection = self._create_connection()
                    self._created_connections += 1
                    logger.debug(f"Created new connection (total: {self._created_connections})")
                else:
                    # If we've hit the limit, wait for a connection
                    connection = self._pool.get(block=True)
        
        try:
            yield connection
        finally:
            try:
                # Reset the connection state
                connection.rollback()
                # Put the connection back in the pool
                self._pool.put(connection)
            except Exception as e:
                logger.error(f"Error returning connection to pool: {e}")
                # If we can't return it to the pool, close it
                try:
                    connection.close()
                    with self._lock:
                        self._created_connections -= 1
                except Exception:
                    pass
