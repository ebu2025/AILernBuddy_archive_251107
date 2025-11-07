import json

import app
import db
from bloom_levels import BLOOM_LEVELS


def test_store_feedback_and_summary(temp_db):
    body = app.FeedbackBody(user_id="user-1", answer_id="ans-1", rating="up", comment="Great", confidence=0.9)
    response = app.receive_feedback(body)
    assert response["status"] == "success"
    summary = response["summary"]
    assert summary["total"] == 1
    assert summary["ratings"]["up"] == 1
    assert summary["average_confidence"] == 0.9

    entries = db.list_feedback(answer_id="ans-1")
    assert entries
    assert entries[0]["comment"] == "Great"


def test_chat_records_diagnosis(monkeypatch, temp_db):
    db.upsert_subject("bpmn", "BPMN", "bpmn", "Process modelling")
    sequence = BLOOM_LEVELS.sequence()
    bloom_level = sequence[0] if sequence else BLOOM_LEVELS.lowest_level()

    payload = {
        "answer": "Here is your feedback",
        "assessment_result": {
            "user_id": "learner-1",
            "domain": "bpmn",
            "item_id": "activity-1",
            "bloom_level": bloom_level,
            "response": "Solution",
            "score": 0.75,
            "rubric_criteria": [],
            "model_version": "model-x",
            "prompt_version": "chat.v1",
            "diagnosis": "procedural",
        },
    }

    monkeypatch.setattr(
        app,
        "generate_with_continue",
        lambda *args, **kwargs: "Answer\n" + json.dumps(payload),
    )

    body = app.ChatBody(user_id="learner-1", topic="bpmn", text="Check this")
    app.chat(body)

    attempts = db.list_recent_quiz_attempts("learner-1", "bpmn", limit=1)
    assert attempts
    assert attempts[0]["diagnosis"] == "procedural"
