import json

import pytest

import app


@pytest.fixture
def stub_db(monkeypatch):
    recorded = {"prompts": [], "logs": [], "ops": []}

    monkeypatch.setattr(app.db, "ensure_user", lambda user_id: None)
    monkeypatch.setattr(app.db, "get_prompts_for_topic", lambda topic, limit=2: [])
    monkeypatch.setattr(
        app.LEARNING_PATH_MANAGER,
        "get_state",
        lambda user_id, subject_id: {"levels": {app._LOWEST_BLOOM_LEVEL: 0.5}, "history": [], "current_level": app._LOWEST_BLOOM_LEVEL},
    )

    def _add_prompt(topic, prompt_text, source=""):
        recorded["prompts"].append((topic, prompt_text, source))

    def _log_journey(user_id, op, payload):
        recorded["logs"].append((user_id, op, payload))

    def _record_chat_ops(
        user_id,
        topic,
        question,
        answer,
        response_json,
        applied_ops,
        pending_ops,
        *,
        raw_response=None,
    ):
        recorded["ops"].append(
            {
                "user_id": user_id,
                "topic": topic,
                "question": question,
                "answer": answer,
                "raw_response": raw_response,
                "response_json": response_json,
                "applied_ops": list(applied_ops or []),
                "pending_ops": list(pending_ops or []),
            }
        )

    monkeypatch.setattr(app.db, "add_prompt", _add_prompt)
    monkeypatch.setattr(app.db, "log_journey_update", _log_journey)
    monkeypatch.setattr(app.db, "record_chat_ops", _record_chat_ops)

    return recorded


def test_chat_applies_ops_and_hides_json(monkeypatch, stub_db):
    response_text = (
        "## Heading\n"
        "- First point\n"
        "- Second point\n\n"
        "{\"db_ops\": [{\"op\": \"add_prompt\", \"payload\": {\"topic\": \"mathematics\", \"prompt_text\": \"New idea\"}}]}"
    )

    monkeypatch.setattr(
        app,
        "generate_with_continue",
        lambda *args, **kwargs: response_text,
    )

    body = app.ChatBody(user_id="learner", topic="mathematics", text="Question?")
    result = app.chat(body)

    assert result["answer_text"] == "## Heading\n- First point\n- Second point"
    assert "{" not in result["answer_text"]
    assert result["pending_ops"] == []
    assert len(result["applied_ops"]) == 1
    assert result["applied_ops"][0]["op"] == "add_prompt"
    assert stub_db["prompts"] == [("mathematics", "New idea", "generated")]
    assert stub_db["logs"] == [("learner", "add_prompt", {"topic": "mathematics", "prompt_text": "New idea"})]
    assert len(stub_db["ops"]) == 1
    entry = stub_db["ops"][0]
    assert entry["user_id"] == "learner"
    assert entry["topic"] == "mathematics"
    assert entry["question"] == "Question?"
    assert entry["answer"] == "## Heading\n- First point\n- Second point"
    assert entry["raw_response"] == response_text
    assert entry["response_json"]["db_ops"][0]["payload"]["prompt_text"] == "New idea"
    assert entry["applied_ops"][0]["op"] == "add_prompt"
    assert entry["pending_ops"] == []


def test_chat_uses_answer_field_when_present(monkeypatch, stub_db):
    json_only = '{"answer": "## Result\\n**Important**", "db_ops": []}'

    monkeypatch.setattr(
        app,
        "generate_with_continue",
        lambda *args, **kwargs: json_only,
    )

    body = app.ChatBody(user_id="learner", topic="business_process", text="Hello")
    result = app.chat(body)

    assert result["answer_text"] == "## Result\n**Important**"
    assert result["applied_ops"] == []
    assert result["pending_ops"] == []
    assert stub_db["prompts"] == []
    assert stub_db["logs"] == []
    assert len(stub_db["ops"]) == 1
    entry = stub_db["ops"][0]
    assert entry["response_json"]["answer"] == "## Result\n**Important**"
    assert entry["raw_response"] == json_only
    assert entry["applied_ops"] == []
    assert entry["pending_ops"] == []


def test_chat_records_pending_ops_when_review_mode(monkeypatch, stub_db):
    response_text = (
        "All right!\n"
        "{\"db_ops\": [{\"op\": \"add_prompt\", \"payload\": {\"topic\": \"language\", \"prompt_text\": \"Test\"}}]}"
    )

    monkeypatch.setattr(
        app,
        "generate_with_continue",
        lambda *args, **kwargs: response_text,
    )

    body = app.ChatBody(user_id="learner", topic="language", text="Question", apply_mode="review")
    result = app.chat(body)

    assert result["applied_ops"] == []
    assert len(result["pending_ops"]) == 1
    assert result["pending_ops"][0]["op"] == "add_prompt"

    assert len(stub_db["ops"]) == 1
    entry = stub_db["ops"][0]
    assert entry["applied_ops"] == []
    assert len(entry["pending_ops"]) == 1
    assert entry["pending_ops"][0]["payload"]["prompt_text"] == "Test"


def test_db_chat_ops_deserializes(monkeypatch):
    payload = {
        "answer": "Hi",
        "db_ops": [{"op": "add_prompt", "payload": {"topic": "math", "prompt_text": "X"}}],
    }

    monkeypatch.setattr(
        app.db,
        "list_chat_ops",
        lambda user_id=None, limit=100: [
            {
                "id": 1,
                "user_id": "learner",
                "topic": "math",
                "question": "Q",
                "answer": "A",
                "response_json": payload,
                "applied_ops": [{"op": "add_prompt"}],
                "pending_ops": [],
                "created_at": "2024-01-01 12:00:00",
            }
        ],
    )

    data = app.db_chat_ops(user_id="learner", limit=10)
    assert len(data) == 1
    row = data[0]
    assert row["response_json"]["answer"] == "Hi"
    assert row["applied_ops"][0]["op"] == "add_prompt"
