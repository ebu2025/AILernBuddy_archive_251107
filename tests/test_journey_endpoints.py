import asyncio
import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


def _post_json(path: str, payload: dict) -> tuple[int, dict]:
    async def _call():
        body = json.dumps(payload).encode("utf-8")
        received_once = False

        async def receive():
            nonlocal received_once
            if not received_once:
                received_once = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        messages = []

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "scheme": "http",
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("testclient", 12345),
            "server": ("testserver", 80),
            "state": {},
        }

        await app.app(scope, receive, send)
        return messages

    messages = asyncio.run(_call())
    status = 500
    body_bytes = b""

    for message in messages:
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            body_bytes += message.get("body", b"")

    data = json.loads(body_bytes.decode("utf-8") or "{}")
    return status, data


class JourneyEndpointsTests(unittest.TestCase):
    def test_session_end_rejects_foreign_user(self):
        session = {"session_id": "owner:abc", "user_id": "owner"}
        with patch("app.journey_tracker.get_session", return_value=session) as get_session, patch(
            "app.journey_tracker.complete_session"
        ) as complete_session:
            status, payload = _post_json(
                "/journey/session/end",
                {"user_id": "intruder", "session_id": "owner:abc"},
            )

        self.assertEqual(status, 403)
        self.assertEqual(payload["detail"], "session does not belong to user")
        get_session.assert_called_once_with("owner:abc")
        complete_session.assert_not_called()

    def test_session_end_updates_authorized_session(self):
        session = {"session_id": "owner:abc", "user_id": "owner"}
        with patch("app.journey_tracker.get_session", return_value=session), patch(
            "app.journey_tracker.complete_session", return_value={"session_id": "owner:abc", "user_id": "owner"}
        ) as complete_session:
            status, payload = _post_json(
                "/journey/session/end",
                {"user_id": "owner", "session_id": "owner:abc", "summary": {"result": "ok"}},
            )

        self.assertEqual(status, 200)
        self.assertEqual(payload["session_id"], "owner:abc")
        complete_session.assert_called_once()
        _, kwargs = complete_session.call_args
        self.assertEqual(kwargs["user_id"], "owner")
        self.assertEqual(kwargs["session_id"], "owner:abc")

    def test_session_end_returns_not_found_for_unknown_session(self):
        with patch("app.journey_tracker.get_session", return_value=None) as get_session, patch(
            "app.journey_tracker.complete_session"
        ) as complete_session:
            status, payload = _post_json(
                "/journey/session/end",
                {"user_id": "owner", "session_id": "owner:missing"},
            )

        self.assertEqual(status, 404)
        self.assertEqual(payload["detail"], "session not found")
        get_session.assert_called_once_with("owner:missing")
        complete_session.assert_not_called()

    def test_event_endpoint_validates_session_user(self):
        session = {"session_id": "owner:abc", "user_id": "owner"}
        with patch("app.db.ensure_user"), patch("app.journey_tracker.get_session", return_value=session), patch(
            "app.journey_tracker.record_event"
        ) as record_event:
            status, payload = _post_json(
                "/journey/event",
                {
                    "user_id": "intruder",
                    "session_id": "owner:abc",
                    "event_type": "reflection",
                },
            )

        self.assertEqual(status, 403)
        self.assertEqual(payload["detail"], "session does not belong to user")
        record_event.assert_not_called()

    def test_event_endpoint_defaults_subject_from_session(self):
        session = {"session_id": "owner:abc", "user_id": "owner", "subject_id": "math"}
        with patch("app.db.ensure_user"), patch("app.journey_tracker.get_session", return_value=session), patch(
            "app.journey_tracker.record_event", return_value={"session_id": "owner:abc"}
        ) as record_event:
            status, payload = _post_json(
                "/journey/event",
                {
                    "user_id": "owner",
                    "session_id": "owner:abc",
                    "event_type": "reflection",
                },
            )

        self.assertEqual(status, 200)
        record_event.assert_called_once()
        kwargs = record_event.call_args.kwargs
        self.assertEqual(kwargs["subject_id"], "math")
        self.assertIn("details", kwargs)
        self.assertEqual(kwargs["details"]["skill_id"], "math")
        self.assertEqual(kwargs["details"]["competency_id"], "math")
        self.assertEqual(kwargs["details"]["outcome"], "unknown")

    def test_event_endpoint_rejects_subject_mismatch(self):
        session = {"session_id": "owner:abc", "user_id": "owner", "subject_id": "math"}
        with patch("app.db.ensure_user"), patch("app.journey_tracker.get_session", return_value=session), patch(
            "app.journey_tracker.record_event"
        ) as record_event:
            status, payload = _post_json(
                "/journey/event",
                {
                    "user_id": "owner",
                    "session_id": "owner:abc",
                    "subject_id": "science",
                    "event_type": "reflection",
                },
            )

        self.assertEqual(status, 400)
        self.assertEqual(payload["detail"], "subject mismatch with session")
        record_event.assert_not_called()

    def test_event_endpoint_returns_not_found_for_unknown_session(self):
        with patch("app.db.ensure_user"), patch("app.journey_tracker.get_session", return_value=None) as get_session, patch(
            "app.journey_tracker.record_event"
        ) as record_event:
            status, payload = _post_json(
                "/journey/event",
                {
                    "user_id": "owner",
                    "session_id": "owner:missing",
                    "event_type": "reflection",
                },
            )

        self.assertEqual(status, 404)
        self.assertEqual(payload["detail"], "session not found")
        get_session.assert_called_once_with("owner:missing")
        record_event.assert_not_called()

    def test_event_endpoint_requires_skill_or_competency(self):
        with patch("app.db.ensure_user"):
            status, payload = _post_json(
                "/journey/event",
                {
                    "user_id": "learner",
                    "event_type": "reflection",
                    "details": {},
                },
            )

        self.assertEqual(status, 400)
        self.assertEqual(payload["detail"], "skill metadata required for event")

    def test_build_event_log_injects_metadata(self):
        entry = {
            "id": 1,
            "op": "standalone_event",
            "payload": {
                "event_type": "practice",
                "subject_id": "math.algebra",
                "score": 0.9,
                "details": {"success": True},
                "recorded_at": "2024-03-01T10:00:00Z",
            },
            "created_at": "2024-03-01T10:00:00Z",
        }
        with patch("app.db.list_journey", return_value=[entry]):
            events = app._build_event_log("learner", "math")

        self.assertEqual(len(events), 1)
        metadata = events[0]["metadata"]
        self.assertEqual(metadata["skill_id"], "math.algebra")
        self.assertEqual(metadata["competency_id"], "math.algebra")
        self.assertEqual(metadata["outcome"], "success")

    def test_journey_diagnostic_start_returns_calibration(self):
        items = [
            {"id": "easy", "skill": "topic", "difficulty": -0.4, "body": "Easy"},
            {"id": "core", "skill": "topic", "difficulty": 0.0, "body": "Core"},
        ]
        calibration = {
            "theta_before": 0.0,
            "theta_after": -0.6,
            "confidence_before": 0.5,
            "confidence_after": 0.65,
            "confidence_growth": 0.15,
            "penalties_applied": 2,
            "penalty_trace": [],
            "placement_band": "intro",
            "bloom_lock": ["K1", "K2"],
        }

        with patch("app.db.ensure_user") as ensure_user, patch("app.db.log_journey_update") as log_update, patch(
            "app.journey.select_calibration_items", return_value=items
        ) as select_items, patch(
            "app.journey.prepare_diagnostic_calibration", return_value=calibration
        ) as prepare_calibration, patch(
            "app.journey_tracker.start_session",
            return_value={"session_id": "user-3:ghi", "subject_id": "topic"},
        ) as start_session, patch("app.journey_tracker.record_event") as record_event:
            status, payload = _post_json(
                "/journey/diagnostic/start",
                {"user_id": "user-3", "subject_id": "topic", "limit": 4},
            )

        self.assertEqual(status, 200)
        ensure_user.assert_called_once_with("user-3")
        select_items.assert_called_once()
        args, kwargs = select_items.call_args
        self.assertEqual(kwargs.get("user_id"), "user-3")
        self.assertEqual(kwargs.get("limit"), 4)
        prepare_calibration.assert_called_once()
        start_session.assert_called_once()
        self.assertEqual(record_event.call_count, 2)
        log_update.assert_called_once()
        self.assertIn("diagnostic_items", payload)
        self.assertIn("calibration", payload)
        self.assertEqual(payload["journey_session"]["session_id"], "user-3:ghi")
        self.assertEqual(payload["calibration"]["placement_band"], "intro")


if __name__ == "__main__":
    unittest.main()
