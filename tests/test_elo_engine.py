import importlib
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import db
import engines.elo as elo_module


class EloEngineDbTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._prev_db_path = os.environ.get("DB_PATH")
        os.environ["DB_PATH"] = os.path.join(self._tmpdir.name, "test.db")

        importlib.reload(db)
        db.init()
        importlib.reload(elo_module)

        self.engine = elo_module.EloEngine(k=0.5)

    def tearDown(self):
        if self._prev_db_path is not None:
            os.environ["DB_PATH"] = self._prev_db_path
        else:
            os.environ.pop("DB_PATH", None)

        importlib.reload(db)
        importlib.reload(elo_module)
        self._tmpdir.cleanup()

    def test_update_persists_theta_and_reuses_existing_value(self):
        db.upsert_item("item-1", "algebra", difficulty=0.0, body="What is 2+2?")

        theta_before = db.get_theta("alice", "algebra")
        self.assertEqual(theta_before, 0.0)

        beta = 0.0
        result_first = self.engine.update("alice", "algebra", "item-1", correct=1)

        expected_p_first = 1.0 / (1.0 + math.exp(-(theta_before - beta)))
        expected_theta_first = theta_before + 0.5 * (1 - expected_p_first)

        self.assertAlmostEqual(result_first["theta_before"], 0.0)
        self.assertAlmostEqual(result_first["theta_after"], expected_theta_first)
        self.assertAlmostEqual(result_first["confidence_before"], expected_p_first)
        self.assertAlmostEqual(result_first["confidence_after"], self.engine.predict_success(expected_theta_first, beta))
        self.assertIn(result_first["placement_band"], {"intro", "core", "stretch"})
        self.assertAlmostEqual(db.get_theta("alice", "algebra"), expected_theta_first)

        result_second = self.engine.update("alice", "algebra", "item-1", correct=0)

        expected_p_second = 1.0 / (1.0 + math.exp(-(expected_theta_first - beta)))
        expected_theta_second = expected_theta_first + 0.5 * (0 - expected_p_second)

        self.assertAlmostEqual(result_second["theta_before"], expected_theta_first)
        self.assertAlmostEqual(result_second["theta_after"], expected_theta_second)
        self.assertAlmostEqual(
            result_second["confidence_before"], self.engine.predict_success(expected_theta_first, beta)
        )
        self.assertAlmostEqual(
            result_second["confidence_after"], self.engine.predict_success(expected_theta_second, beta)
        )
        self.assertAlmostEqual(db.get_theta("alice", "algebra"), expected_theta_second)

        mastery_rows = db.list_mastery(user_id="alice")
        self.assertEqual(len(mastery_rows), 1)


if __name__ == "__main__":
    unittest.main()
