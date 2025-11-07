import logging

import pytest

import tutor
from engines import progression


def test_progression_retry_question_softens_by_diagnosis(caplog):
    base = 1100.0

    with caplog.at_level(logging.INFO, logger="engines.progression"):
        conceptual = progression.retry_question(base, "conceptual")
        procedural = progression.retry_question(base, "procedural")
        careless = progression.retry_question(base, "careless")

    assert conceptual["target_elo"] < base
    assert procedural["target_elo"] < base
    assert careless["target_elo"] <= base
    assert conceptual["target_elo"] < procedural["target_elo"] <= careless["target_elo"]

    logged_messages = [rec.getMessage() for rec in caplog.records]
    assert any("target_elo adjusted" in message for message in logged_messages)


def test_tutor_retry_question_logs_target(monkeypatch):
    calls: list[tuple[str, str, dict]] = []

    class DummyDb:
        @staticmethod
        def log_journey_update(user_id, op, payload):
            calls.append((user_id, op, payload))

    result = tutor.retry_question(
        "learner-1",
        "algebra",
        "activity-7",
        current_target=1050.0,
        diagnosis="procedural",
        metadata={"source": "unit-test"},
        db_module=DummyDb(),
    )

    assert calls, "expected retry decision to be logged"
    user_id, op, payload = calls[0]
    assert user_id == "learner-1"
    assert op == "retry_question"
    assert pytest.approx(payload["target_elo_before"]) == 1050.0
    assert payload["target_elo_after"] == result["target_elo"]
    assert payload["diagnosis"] == "procedural"
    assert result["delta"] < 0
