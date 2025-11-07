import asyncio
import json
from typing import Optional
from urllib.parse import urlencode

import db
import app
from schemas import AssessmentResult, RubricCriterion


def _make_assessment(user_id: str) -> AssessmentResult:
    return AssessmentResult(
        user_id=user_id,
        domain="math",
        item_id="diagnostic-1",
        bloom_level="K2",
        response="Answer",
        score=0.7,
        rubric_criteria=[RubricCriterion(id="logic", score=0.7)],
        model_version="test",
        prompt_version="p.v1",
        confidence=0.9,
        source="direct",
    )


def test_privacy_export_and_delete(temp_db):
    user_id = "alice"
    db.create_user(user_id, "alice@example.com", "hash", None)
    db.record_privacy_consent(user_id, True, "unit test")
    db.upsert_mastery(user_id, "math.algebra", 0.42)
    db.upsert_bloom_progress(user_id, "math", "K2")
    db.record_llm_metric(user_id, "model-x", "chat.v1", app.CHAT_PROMPT_VARIANT, 1200, 210, 180)
    db.upsert_subject("math", "Mathematics", "math")
    db.log_learning_event(user_id, "math", "quiz", score=0.8, details={"skill": "math.algebra"})
    db.log_journey_update(user_id, "suggest_next_item", {"skill": "math.algebra", "reason": "Practice quadratic functions"})
    db.record_chat_ops(user_id, "math", "Question?", "Answer!", {}, [], [])
    db.upsert_learning_path_state(user_id, "math", {"levels": {"K1": 0.6}, "current_level": "K1", "history": []})
    db.log_learning_path_event(
        user_id,
        "math",
        "K1",
        "promote",
        reason="Test",
        reason_code="unit_test",
        confidence=0.9,
        evidence={"source": "unit"},
    )
    db.store_feedback(user_id, "answer-1", "up", "Great explanation", confidence=0.95)
    db._exec(
        "INSERT INTO xapi_statements(user_id, verb, object_id) VALUES (?,?,?)",
        (user_id, "verb", "obj"),
    )
    db.upsert_module("module-1", "math", "Test module", "K1", "Test description")
    db.upsert_lesson("lesson-1", "module-1", "Test lesson", "Test summary")
    db.upsert_activity("activity-1", "lesson-1", "quiz", "Test activity", target_level="K1")
    db.record_quiz_attempt(user_id, "math", "activity-1", 4, 5)
    db.save_assessment_result(_make_assessment(user_id))

    timestamp = "2024-01-01T00:00:00"
    db._exec(
        """
        INSERT INTO learner_profile (
          user_id, goals_json, preferences_json, history_summary, created_at, updated_at
        ) VALUES (?,?,?,?,?,?)
        """,
        (
            user_id,
            db.json_dumps(["finish unit"]),
            db.json_dumps({"style": "visual"}),
            "Learner history",
            timestamp,
            timestamp,
        ),
    )
    db._exec(
        """
        INSERT INTO learner_priors (
          user_id, skill_id, proficiency, bloom_low, bloom_high, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?)
        """,
        (user_id, "math.algebra", 0.55, "K1", "K3", timestamp, timestamp),
    )
    db._exec(
        """
        INSERT INTO learner_confidence (
          user_id, skill_id, confidence, created_at, updated_at
        ) VALUES (?,?,?,?,?)
        """,
        (user_id, "math.algebra", 0.82, timestamp, timestamp),
    )
    db._exec(
        """
        INSERT INTO learner_misconceptions (
          user_id, skill_id, description, severity, evidence_json, last_seen, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            user_id,
            "math.algebra",
            "Sign confusion",
            "medium",
            db.json_dumps({"notes": "needs review"}),
            timestamp,
            timestamp,
            timestamp,
        ),
    )
    db._exec(
        """
        INSERT INTO eval_pretest_attempts (
          learner_id, topic, score, max_score, metadata
        ) VALUES (?,?,?,?,?)
        """,
        (user_id, "math", 3.0, 5.0, db.json_dumps({"attempt": 1})),
    )
    db._exec(
        """
        INSERT INTO eval_posttest_attempts (
          learner_id, topic, score, max_score, metadata
        ) VALUES (?,?,?,?,?)
        """,
        (user_id, "math", 4.0, 5.0, db.json_dumps({"attempt": 2})),
    )

    export_bundle = app.privacy_export(user_id)
    assert export_bundle["users"]
    assert export_bundle["mastery"]
    assert export_bundle["learning_events"]
    assert export_bundle["journey_log"]
    assert export_bundle["xapi_statements"]
    assert export_bundle["assessment_results"]
    assert export_bundle["bloom_progress"]
    assert export_bundle["bloom_progress_history"]
    assert export_bundle["llm_metrics"]
    assert export_bundle["learning_path_state"]
    assert export_bundle["learning_path_events"]
    assert export_bundle["answer_feedback"]
    assert export_bundle["user_consent"]
    assert export_bundle["learner_profile"][0]["goals_json"] == ["finish unit"]
    assert export_bundle["learner_priors"]
    assert export_bundle["learner_confidence"]
    assert export_bundle["learner_misconceptions"][0]["evidence_json"]["notes"] == "needs review"
    assert export_bundle["eval_pretest_attempts"][0]["metadata"]["attempt"] == 1
    assert export_bundle["eval_posttest_attempts"][0]["metadata"]["attempt"] == 2

    deletion_result = app.privacy_delete(user_id)
    assert deletion_result["deleted"]["users"] == 1

    remaining = db.export_user_data(user_id)
    assert not any(
        remaining.get(key)
        for key in (
            "mastery",
            "learning_events",
            "journey_log",
            "xapi_statements",
            "assessment_results",
            "bloom_progress",
            "bloom_progress_history",
            "llm_metrics",
            "learning_path_state",
            "learning_path_events",
            "answer_feedback",
            "user_consent",
            "learner_profile",
            "learner_priors",
            "learner_confidence",
            "learner_misconceptions",
            "eval_pretest_attempts",
            "eval_posttest_attempts",
        )
    )


def test_privacy_export_requires_token(temp_db):
    app.TOKENS.clear()
    status, payload = _get("/privacy/export", query={"user_id": "alice"})

    assert status == 401
    assert payload["detail"] == "missing or invalid token"


def test_privacy_export_rejects_invalid_token(temp_db):
    app.TOKENS.clear()
    db.create_user("alice", "alice@example.com", app._hash_pw("secret"))

    status, payload = _get(
        "/privacy/export",
        query={"user_id": "alice"},
        headers={"Authorization": "Bearer not-real"},
    )

    assert status == 401
    assert payload["detail"] == "missing or invalid token"


def test_privacy_export_allows_valid_token(temp_db):
    app.TOKENS.clear()

    register_payload = {"user_id": "alice", "password": "secret", "email": "alice@example.com"}
    status_register, _ = _post("/auth/register", register_payload)
    assert status_register == 200

    status_login, login_payload = _post("/auth/login", {"user_id": "alice", "password": "secret"})
    assert status_login == 200
    token = login_payload["token"]

    db.record_privacy_consent("alice", True, "unit test")

    status, payload = _get(
        "/privacy/export",
        query={"user_id": "alice"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert status == 200
    assert payload["users"]


def _prepare_scope(
    method: str,
    path: str,
    *,
    headers: Optional[dict[str, str]] = None,
    query: Optional[dict] = None,
    body: bytes = b"",
) -> tuple[dict, list]:
    raw_headers = [(b"host", b"testserver")]
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode("latin-1"), value.encode("utf-8")))
    if body:
        raw_headers.extend(
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ]
        )
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method.upper(),
        "path": path,
        "raw_path": path.encode("utf-8"),
        "root_path": "",
        "scheme": "http",
        "query_string": urlencode(query or {}, doseq=True).encode("utf-8"),
        "headers": raw_headers,
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "state": {},
    }
    return scope, raw_headers


async def _call_app(
    method: str,
    path: str,
    *,
    payload: Optional[dict] = None,
    query: Optional[dict] = None,
    headers: Optional[dict[str, str]] = None,
) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else b""
    scope, raw_headers = _prepare_scope(method, path, headers=headers, query=query, body=body)
    messages: list[dict] = []

    async def receive() -> dict:
        nonlocal body
        if body:
            chunk, body = body, b""
            return {"type": "http.request", "body": chunk, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    await app.app(scope, receive, send)
    status_code = 500
    response_body = b""
    for message in messages:
        if message.get("type") == "http.response.start":
            status_code = message.get("status", 500)
        elif message.get("type") == "http.response.body":
            response_body += message.get("body", b"")
    content = response_body.decode("utf-8")
    return status_code, json.loads(content or "{}")


def _post(path: str, payload: dict, headers: Optional[dict[str, str]] = None) -> tuple[int, dict]:
    return asyncio.run(_call_app("POST", path, payload=payload, headers=headers))


def _get(path: str, *, query: Optional[dict] = None, headers: Optional[dict[str, str]] = None) -> tuple[int, dict]:
    return asyncio.run(_call_app("GET", path, query=query, headers=headers))
