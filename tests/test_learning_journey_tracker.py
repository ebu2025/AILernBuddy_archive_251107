import importlib
import os
import tempfile
import unittest


class LearningJourneyTrackerTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._prev_db_path = os.environ.get("DB_PATH")
        os.environ["DB_PATH"] = os.path.join(self._tmpdir.name, "journey.db")

        import db  # noqa: F401

        self.db = importlib.reload(db)
        self.db.init()
        self.db.ensure_user("learner-1")

        import journey  # noqa: F401

        self.journey = importlib.reload(journey)
        self.tracker = self.journey.LearningJourneyTracker(db_module=self.db, history_window=200)

    def tearDown(self):
        if self._prev_db_path is None:
            os.environ.pop("DB_PATH", None)
        else:
            os.environ["DB_PATH"] = self._prev_db_path

        import db  # noqa: F401

        importlib.reload(db)
        self._tmpdir.cleanup()

    def test_session_flow_and_timeline(self):
        session = self.tracker.start_session(
            "learner-1",
            "math",
            "lesson",
            metadata={"lesson_id": "lesson-1"},
        )
        self.assertIn("session_id", session)
        event = self.tracker.record_event(
            "learner-1",
            "math",
            "practice",
            lesson_id="lesson-1",
            score=0.85,
            details={"success": True},
            session_id=session["session_id"],
        )
        self.assertEqual(event["event_type"], "practice")
        self.assertEqual(event["details"].get("success"), True)
        self.assertEqual(event["session_id"], session["session_id"])
        self.assertEqual(event["details"].get("skill_id"), "math")
        self.assertEqual(event["details"].get("competency_id"), "math")
        self.assertEqual(event["details"].get("outcome"), "success")
        self.assertEqual(event.get("skill_id"), "math")
        self.assertEqual(event.get("outcome"), "success")

        closed = self.tracker.complete_session("learner-1", session["session_id"], summary={"result": "completed"})
        self.assertIsNotNone(closed)
        self.assertEqual(closed["summary"]["result"], "completed")

        self.tracker.record_event(
            "learner-1",
            "math",
            "reflection",
            details={"note": "needs review"},
        )

        timeline = self.tracker.get_timeline("learner-1")
        self.assertEqual(timeline["summary"]["total_sessions"], 1)
        self.assertEqual(timeline["summary"]["events_in_sessions"], 1)
        self.assertEqual(timeline["summary"]["loose_events"], 1)
        self.assertIn("total_duration_seconds", timeline["summary"])
        self.assertIn("skill_area_domain_counts", timeline["summary"])
        self.assertIn("insights", timeline)
        self.assertIn("skill_areas", timeline["insights"])
        self.assertIn("nudges", timeline)
        self.assertEqual(timeline["nudges"], [])
        self.assertEqual(len(timeline["sessions"][0]["events"]), 1)
        self.assertIn("insights", timeline["sessions"][0])
        session_insights = timeline["sessions"][0]["insights"]
        self.assertIn("duration_seconds", session_insights)
        self.assertIn("bloom", session_insights)
        self.assertIn("streaks", session_insights)
        self.assertIn("current_success_streak", session_insights["streaks"])
        self.assertEqual(timeline["loose_events"][0]["event_type"], "reflection")

    def test_complete_session_rejects_mismatched_user(self):
        session = self.tracker.start_session("learner-1", "math", "lesson")
        result = self.tracker.complete_session("intruder", session["session_id"])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
