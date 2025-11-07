import copy
import unittest
from unittest.mock import patch

import journey
import learning_path as learning_path_module
from learning_path import AdaptiveLearningPathManager


class _ColdStartStubDb:
    def __init__(self) -> None:
        self.theta_store: dict[tuple[str, str], float] = {}
        self.items: list[dict[str, object]] = []
        self.item_bank_rows: list[dict[str, object]] = []
        self.progress_snapshot: dict[str, object] | None = None
        self.learning_path_state: dict[tuple[str, str], dict[str, object]] = {}
        self.learning_path_events: list[dict[str, object]] = []
        self.bloom_progress_updates: list[dict[str, object]] = []

    def get_theta(self, user_id: str, skill: str) -> float:
        return self.theta_store.get((user_id, skill), 0.0)

    def list_items(self, skill: str | None = None, limit: int = 10):
        return [dict(row) for row in self.items[:limit]]

    def list_item_bank(self, skill_id: str | None = None, limit: int = 10):
        return [dict(row) for row in self.item_bank_rows[:limit]]

    def get_user_progress(self, user_id: str, subject_id: str):
        if self.progress_snapshot is None:
            return None
        return dict(self.progress_snapshot)

    def get_learning_path_state(self, user_id: str, subject_id: str):
        state = self.learning_path_state.get((user_id, subject_id))
        if state is None:
            return None
        return copy.deepcopy(state)

    def upsert_learning_path_state(self, user_id: str, subject_id: str, state: dict[str, object]):
        self.learning_path_state[(user_id, subject_id)] = copy.deepcopy(state)

    def log_learning_path_event(
        self,
        user_id: str,
        subject_id: str,
        bloom_level: str,
        action: str,
        *,
        reason: str | None = None,
        reason_code: str | None = None,
        confidence: float | None = None,
        evidence: dict[str, object] | None = None,
    ) -> None:
        self.learning_path_events.append(
            {
                "user_id": user_id,
                "subject_id": subject_id,
                "bloom_level": bloom_level,
                "action": action,
                "reason": reason,
                "reason_code": reason_code,
                "confidence": confidence,
                "evidence": evidence,
            }
        )

    def upsert_bloom_progress(
        self,
        user_id: str,
        topic: str,
        current_level: str,
        *,
        reason: str | None = None,
        average_score: float | None = None,
        attempts_considered: int | None = None,
        k_level: str | None = None,
    ) -> None:
        self.bloom_progress_updates.append(
            {
                "user_id": user_id,
                "topic": topic,
                "current_level": current_level,
                "reason": reason,
                "average_score": average_score,
                "attempts_considered": attempts_considered,
                "k_level": k_level,
            }
        )


class ColdStartPolicyTests(unittest.TestCase):
    def test_initial_banding_spans_intro_core_and_stretch(self):
        stub = _ColdStartStubDb()
        user_id, topic = "learner", "algebra"
        stub.theta_store[(user_id, topic)] = 0.7
        stub.items = [
            {"id": "intro", "skill": topic, "difficulty": -0.4, "body": "easy"},
            {"id": "core", "skill": topic, "difficulty": 0.1, "body": "core"},
            {"id": "stretch", "skill": topic, "difficulty": 0.5, "body": "hard"},
            {"id": "reserve", "skill": topic, "difficulty": 0.3, "body": "backup"},
        ]

        selected = journey.select_calibration_items(
            topic,
            limit=3,
            user_id=user_id,
            db_module=stub,
            item_bank={},
            elo_engine=journey.EloEngine(),
        )

        self.assertEqual(len(selected), 3)
        bands = {entry["band"] for entry in selected}
        self.assertSetEqual(bands, {0, 1, 2})
        for entry in selected:
            self.assertEqual(
                entry["band"], journey._band_for_difficulty(entry.get("difficulty"))
            )

    def test_confidence_interval_controls_probability_span(self):
        stub = _ColdStartStubDb()
        user_id, topic = "learner", "topic"
        stub.theta_store[(user_id, topic)] = 0.0
        stub.items = [
            {"id": f"item-{idx}", "skill": topic, "difficulty": diff, "body": f"Body {idx}"}
            for idx, diff in enumerate([-1.4, -0.7, -0.5, -0.3, 0.0, 0.2, 0.6])
        ]
        stub.progress_snapshot = {
            "user_id": user_id,
            "subject_id": topic,
            "current_level": "K1",
            "confidence": 0.62,
            "target_probability": 0.65,
            "ci_width": 0.5,
        }

        engine = journey.EloEngine()
        narrow = journey.select_calibration_items(
            topic,
            limit=3,
            user_id=user_id,
            db_module=stub,
            item_bank={},
            elo_engine=engine,
        )
        theta = stub.get_theta(user_id, topic)
        narrow_probs = [
            engine.predict_success(theta, entry["difficulty"])
            for entry in narrow
            if entry.get("difficulty") is not None
        ]
        self.assertTrue(narrow_probs)
        narrow_span = max(narrow_probs) - min(narrow_probs)

        stub.progress_snapshot["ci_width"] = 0.18
        wide = journey.select_calibration_items(
            topic,
            limit=3,
            user_id=user_id,
            db_module=stub,
            item_bank={},
            elo_engine=engine,
        )
        wide_probs = [
            engine.predict_success(theta, entry["difficulty"])
            for entry in wide
            if entry.get("difficulty") is not None
        ]
        self.assertTrue(wide_probs)
        wide_span = max(wide_probs) - min(wide_probs)

        self.assertGreater(wide_span, narrow_span)
        self.assertTrue(any(prob >= 0.75 for prob in wide_probs))
        self.assertTrue(any(prob <= 0.5 for prob in wide_probs))

    def test_low_confidence_blocks_promotion(self):
        stub = _ColdStartStubDb()
        manager = AdaptiveLearningPathManager()
        sequence = manager._sequence
        if len(sequence) < 2:
            self.skipTest("Bloom sequence requires at least two levels")

        user_id, subject_id = "learner", "algebra"
        current_level = sequence[0]
        stub.learning_path_state[(user_id, subject_id)] = {
            "levels": {level: (0.78 if level == current_level else 0.0) for level in sequence},
            "current_level": current_level,
        }

        with patch.object(learning_path_module, "db", stub):
            recommendation = manager.update_learning_path(
                user_id=user_id,
                subject_id=subject_id,
                bloom_level=current_level,
                correct=True,
                confidence=0.55,
            )

        self.assertEqual(recommendation.action, "hold")
        self.assertEqual(recommendation.recommended_level, current_level)
        self.assertTrue(
            any(event["reason_code"] == "hold_low_confidence" for event in stub.learning_path_events)
        )


if __name__ == "__main__":
    unittest.main()
