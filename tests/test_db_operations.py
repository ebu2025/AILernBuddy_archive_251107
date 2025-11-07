"""Test cases for db operations."""

import pytest
import json
from datetime import datetime, timezone

from db import (
    _conn,
    _exec,
    log_intervention,
    get_user_profile,
)

def test_intervention_logging():
    """Test logging and retrieving intervention records."""
    # Setup
    with _conn() as con:
        con.execute("DROP TABLE IF EXISTS interventions")
    
    # Test data
    user_id = "test_user"
    intervention_type = "struggle"
    confidence = 0.85
    context = {"recent_scores": [0.3, 0.4, 0.35]}
    intervention = {
        "type": "support",
        "message": "Need help?",
        "suggestions": ["Review basics", "Try practice problems"]
    }
    
    # Log intervention
    log_intervention(
        user_id=user_id,
        intervention_type=intervention_type,
        confidence=confidence,
        context=context,
        intervention=intervention
    )
    
    # Verify logging
    with _conn() as con:
        rows = con.execute("SELECT * FROM interventions WHERE user_id = ?", [user_id]).fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["user_id"] == user_id
        assert row["intervention_type"] == intervention_type
        assert row["confidence"] == confidence
        assert json.loads(row["context"]) == context
        assert json.loads(row["intervention"]) == intervention

def test_get_user_profile():
    """Test retrieving user profile."""
    # Setup
    with _conn() as con:
        con.execute("DROP TABLE IF EXISTS users")
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                profile TEXT
            )
        """)
    
    # Test data
    user_id = "test_user"
    profile = {
        "preferred_learning_style": "visual",
        "difficulty_preference": "challenging"
    }
    
    # Insert test user
    _exec(
        "INSERT INTO users (id, profile) VALUES (?, ?)",
        [user_id, json.dumps(profile)]
    )
    
    # Test profile retrieval
    retrieved_profile = get_user_profile(user_id)
    assert retrieved_profile == profile
    
    # Test nonexistent user
    assert get_user_profile("nonexistent") is None
    
    # Test invalid JSON
    _exec(
        "UPDATE users SET profile = ? WHERE id = ?",
        ["{invalid_json", user_id]
    )
    assert get_user_profile(user_id) == {}