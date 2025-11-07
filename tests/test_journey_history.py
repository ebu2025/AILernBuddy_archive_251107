import importlib
import os
import tempfile
import unittest


class JourneyHistoryTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._prev_db_path = os.environ.get("DB_PATH")
        os.environ["DB_PATH"] = os.path.join(self._tmpdir.name, "journey.db")

        import db

        self.db = importlib.reload(db)
        self.db.init()

    def tearDown(self):
        if self._prev_db_path is None:
            os.environ.pop("DB_PATH", None)
        else:
            os.environ["DB_PATH"] = self._prev_db_path

        import db

        importlib.reload(db)
        self._tmpdir.cleanup()

    def test_list_journey_returns_parsed_payload(self):
        self.db.log_journey_update("learner", "module_completed", {"module_id": "mod-1", "confidence": 0.8})

        entries = self.db.list_journey("learner")
        self.assertEqual(len(entries), 1)
        payload = entries[0]["payload"]
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["module_id"], "mod-1")
        self.assertAlmostEqual(payload["confidence"], 0.8)

    def test_list_journey_keeps_plain_payload_if_not_json(self):
        self.db._exec(
            "INSERT INTO journey_log(user_id, op, payload) VALUES (?,?,?)",
            ("learner", "note", "not-json"),
        )

        entries = self.db.list_journey("learner")
        self.assertEqual(entries[0]["payload"], "not-json")

    def test_list_chat_ops_decodes_json_fields(self):
        self.db.record_chat_ops(
            user_id="learner",
            topic="fractions",
            question="What is 1/2 + 1/4?",
            answer="The answer is 3/4.",
            response_json=None,
            applied_ops=[{"op": "journey_update", "details": {"lesson": "fractions"}}],
            pending_ops=[],
        )

        entries = self.db.list_chat_ops("learner")
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertIsNone(entry["response_json"])
        self.assertIsInstance(entry["applied_ops"], list)
        self.assertEqual(entry["applied_ops"][0]["op"], "journey_update")
        self.assertEqual(entry["pending_ops"], [])


if __name__ == "__main__":
    unittest.main()
