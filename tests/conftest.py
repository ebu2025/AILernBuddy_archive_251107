import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    import db

    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_path))
    try:
        import xapi

        monkeypatch.setattr(xapi, "DB_PATH", str(db_path))
    except ImportError:
        pass

    # Reset the connection pool for each test
    db._pool = db.SQLiteConnectionPool(str(db_path), max_connections=10)
    # Reset the global pool reference
    db._pool = db.SQLiteConnectionPool(str(db_path), max_connections=10)
    db.init()
    return str(db_path)
