import json

import pytest

from bloom_levels import BLOOM_LEVELS, BloomLevelConfigError

try:
    _K_LEVEL_SEQUENCE = BLOOM_LEVELS.k_level_sequence()
except BloomLevelConfigError:
    sequence = BLOOM_LEVELS.sequence()
    _K_LEVEL_SEQUENCE = sequence[:3] if sequence else ("K1", "K2", "K3")

LOWEST_LEVEL = BLOOM_LEVELS.lowest_level()
SECOND_LEVEL = _K_LEVEL_SEQUENCE[1] if len(_K_LEVEL_SEQUENCE) > 1 else LOWEST_LEVEL
ALLOWED_LEVELS = set(BLOOM_LEVELS.sequence()) or set(_K_LEVEL_SEQUENCE)


@pytest.mark.usefixtures("temp_db")
def test_chat_assessment_triggers_progression(monkeypatch):
    import db
    import app

    db.upsert_subject("bpmn", "BPMN", "bpmn", "Process modelling")
    payload = {
        "answer": "All good",
        "assessment_result": {
            "user_id": "learner-42",
            "domain": "bpmn",
            "item_id": "activity-1",
            "bloom_level": SECOND_LEVEL,
            "response": "Answer",
            "score": 0.92,
            "rubric_criteria": [],
            "model_version": "model-x",
            "prompt_version": "chat.v1",
        },
    }

    monkeypatch.setattr(
        app,
        "generate_with_continue",
        lambda *args, **kwargs: "Answer\n" + json.dumps(payload),
    )

    body = app.ChatBody(user_id="learner-42", topic="bpmn", text="Test")
    app.chat(body)

    progress = db.get_user_progress("learner-42", "bpmn")
    assert progress is not None
    assert progress["current_level"] in ALLOWED_LEVELS

    bloom = db.get_bloom_progress("learner-42", "bpmn")
    assert bloom is not None

    history = db.list_bloom_progress_history(user_id="learner-42", topic="bpmn", limit=5)
    assert history
    assert history[0]["new_level"] == bloom["current_level"]

    attempts = db.list_recent_quiz_attempts("learner-42", "bpmn", limit=1)
    assert attempts
    assert attempts[0]["activity_id"] == "activity-1"

    journey_entries = db.list_journey(user_id="learner-42", limit=10)
    suggestions = [entry for entry in journey_entries if entry.get("op") == "suggest_next_item"]
    assert suggestions, "expected orchestrated recommendation to be logged"
    payload = suggestions[0].get("payload") or {}
    assert payload.get("domain") == "business_process"
    assert payload.get("skill")
    plan = payload.get("plan") or {}
    assert isinstance(plan.get("ordered_nodes"), list)


@pytest.mark.usefixtures("temp_db")
def test_chat_prefers_tracked_level_over_model_hint(monkeypatch):
    import db
    import app

    db.upsert_subject("bpmn", "BPMN", "bpmn", "Process modelling")

    sequence = BLOOM_LEVELS.sequence() or ("K1", "K2")
    tracked_level = sequence[0]
    suggested_level = sequence[-1]
    if suggested_level == tracked_level and len(sequence) > 1:
        suggested_level = sequence[1]

    monkeypatch.setattr(
        app.LEARNING_PATH_MANAGER,
        "get_state",
        lambda *args, **kwargs: {
            "current_level": tracked_level,
            "levels": {tracked_level: 0.55},
        },
    )

    payload = {
        "answer": "All good",
        "bloom_level": suggested_level,
        "db_ops": [],
    }

    monkeypatch.setattr(
        app,
        "generate_with_continue",
        lambda *args, **kwargs: "Answer\n" + json.dumps(payload),
    )

    body = app.ChatBody(user_id="learner-42", topic="bpmn", text="Test")
    result = app.chat(body)

    assert f"Niveau {tracked_level}" in result["explanation"]
