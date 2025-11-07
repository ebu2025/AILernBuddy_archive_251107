import threading

import pytest

import db
import xapi


@pytest.mark.usefixtures("temp_db")
def test_xapi_emit_persists_and_calls_lrs(monkeypatch):
    db.init()

    event = threading.Event()
    calls = []

    async def fake_forward(statement, *, lrs_url, headers, timeout=5.0, max_attempts=3):
        calls.append((lrs_url, statement, headers, timeout, max_attempts))
        event.set()

    monkeypatch.setattr(xapi, "_forward_statement_with_retry", fake_forward)
    monkeypatch.setenv("LRS_URL", "https://example.com/xapi")
    monkeypatch.setenv("LRS_AUTH", "Token abc")

    xapi.emit(
        user_id="alice",
        verb="http://adlnet.gov/expapi/verbs/answered",
        object_id="activity:sample",
        score=0.75,
        success=True,
        response={"detail": "ok"},
        context={"bloom": "K2", "skill": "math.algebra", "confidence": 0.9},
    )

    rows = db._query(
        "SELECT user_id, verb, object_id, score, success, response, context FROM xapi_statements"
    )
    assert len(rows) == 1
    stored = dict(rows[0])
    assert stored["user_id"] == "alice"
    assert stored["verb"].endswith("answered")
    assert pytest.approx(stored["score"], rel=1e-6) == 0.75
    assert stored["success"] == 1

    event.wait(0.5)
    assert calls
    url, payload, headers, timeout, attempts = calls[0]
    assert url == "https://example.com/xapi"
    assert headers["Authorization"] == "Token abc"
    assert headers["Content-Type"] == "application/json"
    assert timeout == 5.0
    assert attempts == 3
    assert payload["verb"]["id"].endswith("answered")
    assert payload["context"]["extensions"]["confidence"] == pytest.approx(0.9)

    monkeypatch.delenv("LRS_URL", raising=False)
    monkeypatch.delenv("LRS_AUTH", raising=False)


@pytest.mark.usefixtures("temp_db")
def test_validate_statement_rejects_unknown_verb(monkeypatch):
    monkeypatch.setenv("APP_BASE_URL", "https://local")
    statement = {
        "actor": {"account": {"homePage": "https://local", "name": "bob"}},
        "verb": {"id": "https://example.com/verbs/custom"},
        "object": {"id": "activity:xyz"},
        "context": {"extensions": {}},
    }
    with pytest.raises(ValueError):
        xapi.validate_statement(statement)
