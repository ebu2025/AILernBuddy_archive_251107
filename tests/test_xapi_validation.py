from __future__ import annotations

import pytest
from fastapi import HTTPException

import db


@pytest.fixture
def app_module(monkeypatch, temp_db):
    monkeypatch.setenv("APP_BASE_URL", "https://test.local")
    monkeypatch.setenv("XAPI_PLATFORM", "AILearnBuddy")

    import app

    monkeypatch.setattr(app.tutor, "ensure_seed_items", lambda: None)
    monkeypatch.setattr(app, "_ensure_rag_ready", lambda force=False: None)

    db._exec("DELETE FROM xapi_statements")
    return app


def test_emit_endpoint_accepts_valid_statement(app_module):
    payload = app_module.XAPIEmitBody(
        user_id="learner-123",
        verb="http://adlnet.gov/expapi/verbs/answered",
        object_id="activity:test-item",
        context={"bloom": "K2"},
    )

    result = app_module.emit_xapi_statement(payload)
    assert result == {"status": "ok"}

    rows = db._query("SELECT verb, object_id FROM xapi_statements")
    assert len(rows) == 1
    stored = dict(rows[0])
    assert stored["verb"] == "http://adlnet.gov/expapi/verbs/answered"
    assert stored["object_id"] == "activity:test-item"


def test_emit_endpoint_rejects_unknown_verb(app_module):
    payload = app_module.XAPIEmitBody(
        user_id="learner-123",
        verb="https://example.com/verbs/custom",
        object_id="activity:test-item",
    )

    with pytest.raises(HTTPException) as excinfo:
        app_module.emit_xapi_statement(payload)

    assert excinfo.value.status_code == 400
    assert "Allowed verbs" in excinfo.value.detail


def test_emit_endpoint_rejects_missing_object_id(app_module):
    payload = app_module.XAPIEmitBody(
        user_id="learner-123",
        verb="http://adlnet.gov/expapi/verbs/answered",
        object_id="  ",
    )

    with pytest.raises(HTTPException) as excinfo:
        app_module.emit_xapi_statement(payload)

    assert excinfo.value.status_code == 400
    assert "object.id must be a non-empty string" in excinfo.value.detail
