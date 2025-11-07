import importlib
import os
import tempfile
import unittest


class JourneyDiagnosticCalibrationTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._prev_db_path = os.environ.get("DB_PATH")
        os.environ["DB_PATH"] = os.path.join(self._tmpdir.name, "diagnostic.db")

        import db  # noqa: F401

        self.db = importlib.reload(db)
        self.db.init()
        self.db.ensure_user("learner-1")

        import journey  # noqa: F401

        self.journey = importlib.reload(journey)

        import engines.elo as elo_module  # noqa: F401

        self.elo_module = importlib.reload(elo_module)
        self.elo_engine = self.elo_module.EloEngine(k=0.4)

    def tearDown(self):
        if self._prev_db_path is None:
            os.environ.pop("DB_PATH", None)
        else:
            os.environ["DB_PATH"] = self._prev_db_path

        import db  # noqa: F401

        importlib.reload(db)
        self._tmpdir.cleanup()

    def test_select_calibration_items_widens_difficulty(self):
        diffs = [-0.6, 0.0, 0.65, 0.25]
        for idx, diff in enumerate(diffs):
            self.db.upsert_item(f"item-{idx}", "topic", difficulty=diff, body=f"Body {idx}")

        items = self.journey.select_calibration_items(
            "topic",
            limit=3,
            user_id="learner-1",
            db_module=self.db,
            item_bank={},
            elo_engine=self.elo_engine,
        )
        returned_difficulties = [item.get("difficulty") for item in items]

        self.assertTrue(any(val is not None and val <= -0.2 for val in returned_difficulties))
        self.assertTrue(any(val is not None and -0.2 < val < 0.2 for val in returned_difficulties))
        self.assertTrue(any(val is not None and val >= 0.2 for val in returned_difficulties))

    def test_select_calibration_items_targets_mid_probability(self):
        diffs = [-0.7, -0.4, -0.1, 0.15]
        for idx, diff in enumerate(diffs):
            self.db.upsert_item(f"mid-item-{idx}", "topic", difficulty=diff, body=f"Body {idx}")

        items = self.journey.select_calibration_items(
            "topic",
            limit=3,
            user_id="learner-1",
            db_module=self.db,
            item_bank={},
            elo_engine=self.elo_engine,
        )

        theta = self.db.get_theta("learner-1", "topic")
        predicted = [
            self.elo_engine.predict_success(theta, item["difficulty"])
            for item in items
            if item.get("difficulty") is not None
        ]

        self.assertTrue(predicted)
        for prob in predicted:
            self.assertLessEqual(abs(prob - 0.65), 0.2)

    def test_select_calibration_items_expands_when_confident(self):
        diffs = [-1.2, -0.4, 0.0, 0.6]
        for idx, diff in enumerate(diffs):
            self.db.upsert_item(f"conf-item-{idx}", "topic", difficulty=diff, body=f"Body {idx}")

        self.db.upsert_user_progress("learner-1", "topic", "K1", confidence=0.82)

        items = self.journey.select_calibration_items(
            "topic",
            limit=3,
            user_id="learner-1",
            db_module=self.db,
            item_bank={},
            elo_engine=self.elo_engine,
        )

        theta = self.db.get_theta("learner-1", "topic")
        predicted = [
            self.elo_engine.predict_success(theta, item["difficulty"])
            for item in items
            if item.get("difficulty") is not None
        ]

        self.assertTrue(any(prob >= 0.7 for prob in predicted))
        self.assertTrue(any(prob <= 0.5 for prob in predicted))

    def test_prepare_diagnostic_calibration_updates_confidence_and_bloom_lock(self):
        result = self.journey.prepare_diagnostic_calibration(
            "learner-1",
            "topic",
            db_module=self.db,
            elo_engine=self.elo_engine,
            min_confidence=0.6,
            penalty_step=0.3,
            bloom_lock=("K1", "K2"),
        )

        self.assertGreaterEqual(result["confidence_after"], 0.6)
        self.assertGreater(result["confidence_growth"], 0.0)
        self.assertEqual(result["bloom_lock"], ["K1", "K2"])
        self.assertGreaterEqual(result["penalties_applied"], 1)
        self.assertEqual(result["penalties_applied"], len(result["penalty_trace"]))
        self.assertEqual(result["placement_band"], "intro")

        progress = self.db.get_user_progress("learner-1", "topic")
        self.assertIsNotNone(progress)
        self.assertAlmostEqual(progress["confidence"], result["confidence_after"], places=3)

        bloom = self.db.get_bloom_progress("learner-1", "topic")
        self.assertIsNotNone(bloom)
        self.assertEqual(bloom["current_level"], "K1")

        learner_model = self.db.get_learner_model("learner-1")
        topic_state = next((skill for skill in learner_model.skills if skill.skill_id == "topic"), None)
        self.assertIsNotNone(topic_state)
        self.assertAlmostEqual(topic_state.confidence, result["confidence_after"], places=4)
        self.assertEqual(topic_state.bloom_band.lower, "K1")


if __name__ == "__main__":
    unittest.main()
