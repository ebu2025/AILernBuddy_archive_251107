import math
import unittest

import journey
from schemas import LearnerModel


class _StubDb:
    def __init__(self):
        self.theta_store: dict[tuple[str, str], float] = {}
        self.progress_calls: list[dict[str, object]] = []
        self.bloom_calls: list[tuple[str, str]] = []
        self.updated_model: LearnerModel | None = None
        self.progress_snapshot: dict[str, object] | None = None
        self.items: list[dict[str, object]] = []

    def get_theta(self, user_id: str, subject_id: str) -> float:
        return self.theta_store.get((user_id, subject_id), 0.0)

    def set_theta(self, user_id: str, subject_id: str, value: float) -> None:
        self.theta_store[(user_id, subject_id)] = float(value)

    def upsert_user_progress(
        self,
        user_id: str,
        subject_id: str,
        current_level: str,
        confidence: float | None = None,
        **kwargs,
    ) -> None:
        record = {
            "user_id": user_id,
            "subject_id": subject_id,
            "current_level": current_level,
            "confidence": confidence,
        }
        record.update(kwargs)
        self.progress_calls.append(record)
        self.progress_snapshot = {
            "user_id": user_id,
            "subject_id": subject_id,
            "current_level": current_level,
            **{k: v for k, v in record.items() if k not in {"user_id", "subject_id", "current_level"}},
        }

    def upsert_bloom_progress(self, user_id: str, subject_id: str, current_level: str) -> None:
        self.bloom_calls.append((user_id, subject_id, current_level))

    def get_learner_model(self, user_id: str) -> LearnerModel:
        return LearnerModel(user_id=user_id)

    def update_learner_model(self, model: LearnerModel) -> None:
        self.updated_model = model

    def log_journey_update(self, *_args, **_kwargs) -> None:  # pragma: no cover - logging stub
        pass

    def list_items(self, skill: str | None = None, limit: int = 0):
        return self.items[: limit or None]

    def list_item_bank(self, skill_id: str | None = None, limit: int = 0):
        return []

    def get_user_progress(self, user_id: str, subject_id: str):
        return self.progress_snapshot


class _TrackingEngine:
    def __init__(self, store: _StubDb):
        self._store = store

    @staticmethod
    def predict_success(theta: float, difficulty: float) -> float:
        return 1.0 / (1.0 + math.exp(-(theta - difficulty)))

    @staticmethod
    def placement_band(theta: float) -> str:
        if theta < -0.4:
            return "intro"
        if theta > 0.4:
            return "stretch"
        return "core"

    def apply_penalty(self, user_id: str, skill: str, *, penalty: float = 0.35):
        theta_before = self._store.get_theta(user_id, skill)
        step = abs(float(penalty))
        theta_after = theta_before - step
        self._store.set_theta(user_id, skill, theta_after)
        return {
            "user_id": user_id,
            "skill": skill,
            "theta_before": theta_before,
            "theta_after": theta_after,
            "penalty": -step,
        }


class DiagnosticCalibrationTests(unittest.TestCase):
    def test_prepare_calibration_records_band_and_interval(self):
        stub = _StubDb()
        stub.theta_store[("learner", "topic")] = 0.8
        engine = _TrackingEngine(stub)

        result = journey.prepare_diagnostic_calibration(
            "learner",
            "topic",
            db_module=stub,
            elo_engine=engine,
            bloom_lock=("K1", "K2"),
        )

        self.assertTrue(stub.progress_calls)
        latest_progress = stub.progress_calls[-1]
        self.assertEqual(latest_progress["band_lower"], "K1")
        self.assertEqual(latest_progress["band_upper"], "K2")
        self.assertAlmostEqual(latest_progress["target_probability"], 0.65, places=2)
        self.assertGreaterEqual(latest_progress["ci_width"], 0.25)
        self.assertGreaterEqual(latest_progress["ci_lower"], 0.0)
        self.assertLessEqual(latest_progress["ci_upper"], 1.0)

        self.assertIn("confidence_interval", result)
        interval = result["confidence_interval"]
        self.assertAlmostEqual(interval["center"], 0.65, places=2)
        self.assertAlmostEqual(interval["lower"], latest_progress["ci_lower"], places=4)
        self.assertAlmostEqual(interval["upper"], latest_progress["ci_upper"], places=4)
        self.assertAlmostEqual(interval["width"], latest_progress["ci_width"], places=4)

        self.assertIsNotNone(stub.updated_model)
        skill_state = next(
            (skill for skill in stub.updated_model.skills if skill.skill_id == "topic"),
            None,
        )
        self.assertIsNotNone(skill_state)
        self.assertAlmostEqual(skill_state.target_success_probability, 0.65, places=2)
        self.assertAlmostEqual(
            skill_state.confidence_interval_width,
            interval["width"],
            places=4,
        )

    def test_select_items_widens_only_when_interval_narrows(self):
        stub = _StubDb()
        stub.theta_store[("learner", "topic")] = 0.0
        stub.items = [
            {"id": f"item-{idx}", "skill": "topic", "difficulty": diff, "body": f"Body {idx}"}
            for idx, diff in enumerate([-1.4, -0.7, -0.5, -0.3, 0.2, 1.0])
        ]
        stub.progress_snapshot = {
            "user_id": "learner",
            "subject_id": "topic",
            "current_level": "K1",
            "confidence": 0.6,
            "target_probability": 0.65,
            "ci_width": 0.5,
        }

        engine = journey.EloEngine()

        tight_selection = journey.select_calibration_items(
            "topic",
            limit=3,
            user_id="learner",
            db_module=stub,
            item_bank={},
            elo_engine=engine,
        )
        theta = stub.get_theta("learner", "topic")
        tight_probs = [
            engine.predict_success(theta, item["difficulty"])
            for item in tight_selection
            if item.get("difficulty") is not None
        ]
        self.assertTrue(tight_probs)
        self.assertTrue(all(abs(prob - 0.65) <= 0.15 for prob in tight_probs))

        stub.progress_snapshot["ci_width"] = 0.18
        wide_selection = journey.select_calibration_items(
            "topic",
            limit=3,
            user_id="learner",
            db_module=stub,
            item_bank={},
            elo_engine=engine,
        )
        wide_probs = [
            engine.predict_success(theta, item["difficulty"])
            for item in wide_selection
            if item.get("difficulty") is not None
        ]
        self.assertTrue(wide_probs)
        self.assertGreater(max(wide_probs) - min(wide_probs), max(tight_probs) - min(tight_probs))
        self.assertTrue(any(prob >= 0.75 for prob in wide_probs))
        self.assertTrue(any(prob <= 0.5 for prob in wide_probs))


if __name__ == "__main__":
    unittest.main()
