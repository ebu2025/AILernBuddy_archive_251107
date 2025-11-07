"""SQLite connection pool for better resource management."""
import logging
import sqlite3
import threading
from contextlib import contextmanager
from queue import Empty, Queue
from typing import Generator

logger = logging.getLogger(__name__)

class SQLiteConnectionPool:
    """Thread-safe SQLite connection pool with thread-local connections."""

    def __init__(self, database: str, max_connections: int = 5):
        self.database = database
        self.max_connections = max_connections
        self._local = threading.local()
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
        """Get a thread-local connection from the pool."""
        if not hasattr(self._local, 'connection'):
            with self._lock:
                if self._created_connections < self.max_connections:
                    self._local.connection = self._create_connection()
                    self._created_connections += 1
                    logger.debug(f"Created new thread-local connection (total: {self._created_connections})")
                else:
                    # If we've hit the limit, wait for a connection from the pool
                    try:
                        self._local.connection = self._pool.get(block=True, timeout=5.0)
                    except Empty:
                        raise RuntimeError("Connection pool exhausted")

        connection = self._local.connection
        try:
            yield connection
        finally:
            try:
                # Reset the connection state
                connection.rollback()
            except Exception as e:
                logger.error(f"Error resetting connection state: {e}")
                # If we can't reset, close and create a new one
                try:
                    connection.close()
                    with self._lock:
                        self._created_connections -= 1
                    delattr(self._local, 'connection')
                except Exception:
                    pass
