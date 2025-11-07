import asyncio
import json
from typing import Optional
from unittest.mock import patch
from urllib.parse import urlencode

import app
import db


async def _call_app(method: str, path: str, *, payload: Optional[dict] = None, query: Optional[dict] = None):
    body = b""
    headers = [(b"host", b"testserver")]
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers.extend(
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ]
        )
    query_string = urlencode(query or {}, doseq=True).encode()
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method.upper(),
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "scheme": "http",
        "query_string": query_string,
        "headers": headers,
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "state": {},
    }

    messages = []

    async def receive():
        nonlocal body
        if body:
            chunk, body = body, b""
            return {"type": "http.request", "body": chunk, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    await app.app(scope, receive, send)
    status = 500
    body_bytes = b""
    for message in messages:
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            body_bytes += message.get("body", b"")
    data = json.loads(body_bytes.decode("utf-8") or "{}")
    return status, data


def _post(path: str, payload: dict) -> tuple[int, dict]:
    return asyncio.run(_call_app("POST", path, payload=payload))


def _get(path: str, query: Optional[dict] = None) -> tuple[int, dict]:
    return asyncio.run(_call_app("GET", path, query=query))


def test_pretest_records_instrument_and_session(temp_db):
    session = {"session_id": "learner:s1", "user_id": "learner", "subject_id": "algebra"}
    instrument_payload = {
        "title": "Linear diagnostic",
        "description": "Short pretest",
        "items": [
            {"id": "item-1", "prompt": "Solve for x"},
            {"id": "item-2", "prompt": "Factor"},
        ],
    }
    with patch("app.journey_tracker.get_session", return_value=session) as get_session, patch(
        "app.journey_tracker.record_event"
    ) as record_event:
        status, payload = _post(
            "/eval/pretest",
            {
                "learner_id": "learner",
                "topic": "algebra",
                "score": 3,
                "max_score": 5,
                "session_id": session["session_id"],
                "instrument": instrument_payload,
            },
        )

    assert status == 200
    attempt = payload["attempt"]
    assert attempt["session_id"] == session["session_id"]
    assert attempt["instrument"] is not None
    assert attempt["instrument"]["topic"] == "algebra"
    get_session.assert_called_once_with(session["session_id"])
    assert record_event.call_count == 2

    attached = db.get_eval_session_instrument(session["session_id"], "pretest")
    assert attached is not None
    assert attached["instrument_id"] == attempt["instrument"]["instrument_id"]


def test_pretest_rejects_foreign_session(temp_db):
    session = {"session_id": "owner:s1", "user_id": "owner", "subject_id": "algebra"}
    with patch("app.journey_tracker.get_session", return_value=session):
        status, payload = _post(
            "/eval/pretest",
            {
                "learner_id": "intruder",
                "topic": "algebra",
                "score": 1,
                "max_score": 5,
                "session_id": session["session_id"],
                "instrument": {"items": []},
            },
        )
    assert status == 403
    assert payload["detail"] == "session does not belong to learner"


def test_pretest_rejects_topic_mismatch(temp_db):
    session = {"session_id": "learner:s1", "user_id": "learner", "subject_id": "geometry"}
    with patch("app.journey_tracker.get_session", return_value=session):
        status, payload = _post(
            "/eval/pretest",
            {
                "learner_id": "learner",
                "topic": "algebra",
                "score": 1,
                "max_score": 5,
                "session_id": session["session_id"],
                "instrument": {"items": []},
            },
        )
    assert status == 400
    assert payload["detail"] == "topic mismatch with session subject"


def test_learning_gain_report(temp_db):
    session = {"session_id": "learner:s1", "user_id": "learner", "subject_id": "algebra"}
    instrument_payload = {"items": [{"id": "a"}]}
    with patch("app.journey_tracker.get_session", return_value=session), patch(
        "app.journey_tracker.record_event"
    ) as record_event:
        status_pre, _ = _post(
            "/eval/pretest",
            {
                "learner_id": "learner",
                "topic": "algebra",
                "score": 5,
                "max_score": 10,
                "session_id": session["session_id"],
                "instrument": instrument_payload,
            },
        )
        status_post, _ = _post(
            "/eval/posttest",
            {
                "learner_id": "learner",
                "topic": "algebra",
                "score": 8,
                "max_score": 10,
                "session_id": session["session_id"],
                "instrument_id": "post:manual",
                "instrument": instrument_payload,
            },
        )
    assert status_pre == status_post == 200
    assert record_event.call_count == 4

    status_report, report = _get(
        "/eval/report",
        {"learner_id": "learner", "topic": "algebra"},
    )
    assert status_report == 200
    assert report["overall"]["pair_count"] == 1
    pair = report["pairs"][0]
    assert pair["pre_normalized"] == 0.5
    assert pair["post_normalized"] == 0.8
    assert pair["delta"] == 0.3
    assert pair["normalized_gain"] == 0.6
    assert pair["g"] == 0.6
    assert report["pairs"][0]["pre_instrument"] is not None
    assert report["pairs"][0]["post_instrument"] is not None
    overall = report["overall"]
    assert overall["mean_pre"] == 0.5
    assert overall["mean_post"] == 0.8
    assert overall["mean_delta"] == 0.3
    assert overall["average_normalized_gain"] == 0.6
    assert overall["gain_confidence_interval"] is None


def test_learning_gain_pre_equals_one(temp_db):
    with patch("app.journey_tracker.record_event"):
        status_pre, _ = _post(
            "/eval/pretest",
            {
                "learner_id": "learner",
                "topic": "algebra",
                "score": 10,
                "max_score": 10,
            },
        )
        status_post, _ = _post(
            "/eval/posttest",
            {
                "learner_id": "learner",
                "topic": "algebra",
                "score": 10,
                "max_score": 10,
            },
        )

    assert status_pre == status_post == 200

    status_report, report = _get(
        "/eval/report",
        {"learner_id": "learner", "topic": "algebra"},
    )
    assert status_report == 200
    pair = report["pairs"][0]
    assert pair["pre_normalized"] == 1.0
    assert pair["post_normalized"] == 1.0
    assert pair["delta"] == 0.0
    assert pair["normalized_gain"] is None
    assert pair["g"] is None
    overall = report["overall"]
    assert overall["mean_delta"] == 0.0
    assert overall["average_normalized_gain"] is None


def test_learning_gain_confidence_interval(temp_db):
    learners = [
        {"id": "learner-1", "pre": (5, 10), "post": (8, 10)},
        {"id": "learner-2", "pre": (10, 25), "post": (16, 25)},
    ]
    with patch("app.journey_tracker.record_event"):
        for entry in learners:
            pre_score, pre_max = entry["pre"]
            post_score, post_max = entry["post"]
            status_pre, _ = _post(
                "/eval/pretest",
                {
                    "learner_id": entry["id"],
                    "topic": "algebra",
                    "score": pre_score,
                    "max_score": pre_max,
                },
            )
            status_post, _ = _post(
                "/eval/posttest",
                {
                    "learner_id": entry["id"],
                    "topic": "algebra",
                    "score": post_score,
                    "max_score": post_max,
                },
            )
            assert status_pre == status_post == 200

    status_report, report = _get(
        "/eval/report",
        {"topic": "algebra"},
    )
    assert status_report == 200
    overall = report["overall"]
    assert overall["pair_count"] == 2
    assert overall["average_normalized_gain"] == 0.5
    assert overall["mean_delta"] == 0.27
    ci = overall["gain_confidence_interval"]
    assert ci["confidence_level"] == 0.95
    assert ci["mean"] == 0.5
    assert round(ci["margin"], 4) == ci["margin"]
    assert ci["lower"] < ci["mean"] < ci["upper"]

    pairs = {entry["learner_id"]: entry for entry in report["pairs"]}
    first = pairs["learner-1"]
    assert first["delta"] == 0.3
    assert first["g"] == 0.6
    second = pairs["learner-2"]
    assert second["delta"] == 0.24
    assert second["g"] == 0.4

