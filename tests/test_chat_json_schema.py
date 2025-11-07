"""Tests for chat JSON schema parsing and retry logic.

The expectations mirror ``docs/json/README.md`` so parser, retry, and telemetry
flows remain aligned.
"""

import copy
import json
from typing import Any

import pytest

import app


def _prepare_chat_environment(monkeypatch):
    recorded: dict[str, Any] = {
        "assessments": [],
        "followups": [],
        "lp_state": [],
        "progression": [],
        "chat_ops": None,
        "cleared": [],
    }

    monkeypatch.setattr(app.db, "ensure_user", lambda user_id: None)
    monkeypatch.setattr(app.db, "get_followup_state", lambda user_id, topic: {})
    monkeypatch.setattr(app.db, "get_prompts_for_topic", lambda topic, limit=2: [])
    monkeypatch.setattr(app.db, "add_prompt", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.db, "log_journey_update", lambda *args, **kwargs: None)

    def fake_record_chat_ops(user_id, topic, question, answer, response_json, applied_ops, pending_ops, *, raw_response=None):
        recorded["chat_ops"] = {
            "user_id": user_id,
            "topic": topic,
            "question": question,
            "answer": answer,
            "response_json": response_json,
            "applied_ops": list(applied_ops or []),
            "pending_ops": list(pending_ops or []),
            "raw_response": raw_response,
        }

    monkeypatch.setattr(app.db, "record_chat_ops", fake_record_chat_ops)

    def fake_save_assessment_result(result):
        recorded["assessments"].append(result)

    monkeypatch.setattr(app.db, "save_assessment_result", fake_save_assessment_result)
    monkeypatch.setattr(app.db, "clear_followup_state", lambda user_id, topic: recorded["cleared"].append((user_id, topic)))

    def fake_set_needs_assessment(user_id, topic, needs_assessment, *, microcheck=None):
        recorded["followups"].append(
            {
                "user_id": user_id,
                "topic": topic,
                "needs_assessment": needs_assessment,
                "microcheck": microcheck,
            }
        )

    monkeypatch.setattr(app.db, "set_needs_assessment", fake_set_needs_assessment)

    def fake_upsert_learning_path_state(user_id, topic, state):
        recorded["lp_state"].append({"user_id": user_id, "topic": topic, "state": copy.deepcopy(state)})

    monkeypatch.setattr(app.db, "upsert_learning_path_state", fake_upsert_learning_path_state)

    base_state = {"levels": {app._LOWEST_BLOOM_LEVEL: 0.4}, "current_level": app._LOWEST_BLOOM_LEVEL, "history": []}
    monkeypatch.setattr(app.LEARNING_PATH_MANAGER, "get_state", lambda user_id, topic: copy.deepcopy(base_state))
    monkeypatch.setattr(app.LEARNING_PATH_MANAGER, "update_from_assessment", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "on_assessment_saved", lambda assessment: None)
    monkeypatch.setattr(app, "ensure_progress_record", lambda *args, **kwargs: None)

    def fake_evaluate(user_id, topic, last_attempt_id=None):
        recorded["progression"].append((user_id, topic, last_attempt_id))

    monkeypatch.setattr(app._PROGRESSION_ENGINE, "evaluate", fake_evaluate)

    return recorded


def _valid_response_payload():
    return {
        "answer": "Quick recap.",
        "bloom_level": app._LOWEST_BLOOM_LEVEL,
        "assessment": {
            "item_id": "test-item",
            "bloom_level": app._LOWEST_BLOOM_LEVEL,
            "score": 0.8,
            "confidence": 0.7,
            "response": "Checked key steps.",
            "source": "direct",
        },
        "diagnosis": "conceptual",
        "confidence": 0.6,
        "microcheck_question": "What is 2+2?",
        "microcheck_expected": "4",
        "microcheck_given": None,
        "microcheck_score": None,
        "action": "progression:evaluate",
        "history_update": {"mode": "append", "entries": [{"note": "LLM logged progress", "score": 0.8}]},
        "timestamp": "2024-01-01T00:00:00Z",
        "self_assessment": "I understand.",
        "db_ops": [],
    }


def test_chat_parses_extended_json(monkeypatch):
    recorded = _prepare_chat_environment(monkeypatch)

    response_payload = _valid_response_payload()
    monkeypatch.setattr(
        app,
        "generate_with_continue",
        lambda *args, **kwargs: "Great job!\n" + json.dumps(response_payload),
    )

    body = app.ChatBody(user_id="learner", topic="mathematics", text="How should I proceed?")
    import asyncio
    result = asyncio.run(app.chat(body))

    assert result["answer_text"] == "Quick recap."

    assert recorded["assessments"]
    assessment = recorded["assessments"][0]
    assert pytest.approx(assessment.score) == 0.8
    assert assessment.diagnosis == "conceptual"

    assert recorded["followups"]
    followup_entry = recorded["followups"][0]
    assert followup_entry["needs_assessment"] is True
    assert followup_entry["microcheck"]["answer_key"] == "4"

    assert recorded["lp_state"]
    lp_state = recorded["lp_state"][0]["state"]
    assert lp_state["history"]
    assert lp_state["history"][0]["score"] == 0.8

    assert recorded["progression"] == [("learner", "mathematics", None)]

    chat_record = recorded["chat_ops"]
    assert chat_record["response_json"]["microcheck_state"]["question"] == "What is 2+2?"
    assert chat_record["response_json"]["assessment_result"]["score"] == pytest.approx(0.8)


def test_chat_retries_on_invalid_json(monkeypatch):
    recorded = _prepare_chat_environment(monkeypatch)
    payload = _valid_response_payload()

    outputs = [
        "Preliminary narrative with missing JSON.",
        "Corrected reply\n" + json.dumps(payload),
    ]
    calls: list[dict[str, Any]] = []

    def fake_generate(system, text, max_tokens, **kwargs):
        calls.append(kwargs)
        index = min(len(calls) - 1, len(outputs) - 1)
        return outputs[index]

    monkeypatch.setattr(app, "generate_with_continue", fake_generate)

    body = app.ChatBody(user_id="learner", topic="mathematics", text="How should I proceed?")
    import asyncio
    result = asyncio.run(app.chat(body))

    # Per docs/json/README.md we allow the initial attempt plus a single schema_retry.
    assert len(calls) == 2
    assert calls[0]["extra_context"] == ""
    assert "schema_retry" in calls[1]["path_decisions"]
    assert "failed JSON schema" in calls[1]["extra_context"]
    assert result["answer_text"] == payload["answer"]
    assert recorded["chat_ops"]["raw_response"].startswith("Corrected reply")


def test_chat_returns_error_after_schema_retry_failure(monkeypatch):
    recorded = _prepare_chat_environment(monkeypatch)

    bad_payload = {"answer": "Still broken", "db_ops": "not-a-list"}
    invalid_response = "Narrative\n" + json.dumps(bad_payload)
    calls: list[dict[str, Any]] = []

    def fake_generate(system, text, max_tokens, **kwargs):
        calls.append(kwargs)
        return invalid_response

    monkeypatch.setattr(app, "generate_with_continue", fake_generate)

    body = app.ChatBody(user_id="learner", topic="mathematics", text="How should I proceed?")

    with pytest.raises(app.HTTPException) as excinfo:
        import asyncio
        asyncio.run(app.chat(body))

    assert excinfo.value.status_code == 502
    assert len(calls) == 2
    assert "schema_retry" in calls[1]["path_decisions"]
    assert "failed JSON schema" in calls[1]["extra_context"]
    # The README also documents that we skip chat_ops logging when both attempts fail.
    assert recorded["chat_ops"] is None
