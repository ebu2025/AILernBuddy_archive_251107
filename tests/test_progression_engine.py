import importlib
import os
import tempfile
import unittest

from bloom_levels import BLOOM_LEVELS, BloomLevelConfigError
from learning_path import AdaptiveLearningPathManager
from schemas import (
    AssessmentErrorPattern,
    AssessmentResult,
    AssessmentStepEvaluation,
    RubricCriterion,
)
from engines.intervention_system import LearningInterventionSystem

try:
    _K_LEVEL_SEQUENCE = BLOOM_LEVELS.k_level_sequence()
except BloomLevelConfigError:
    sequence = BLOOM_LEVELS.sequence()
    _K_LEVEL_SEQUENCE = sequence[:3] if sequence else ("K1", "K2", "K3")

LOWEST_LEVEL = BLOOM_LEVELS.lowest_level()
SECOND_LEVEL = _K_LEVEL_SEQUENCE[1] if len(_K_LEVEL_SEQUENCE) > 1 else LOWEST_LEVEL


class ProgressionEngineTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._prev_db_path = os.environ.get("DB_PATH")
        os.environ["DB_PATH"] = os.path.join(self._tmpdir.name, "test.db")

        import db  # noqa: F401

        self.db = importlib.reload(db)
        self.db.init()

        from engines.progression import ProgressionEngine, ensure_progress_record

        self.engine_cls = ProgressionEngine
        self.ensure_progress_record = ensure_progress_record

    def tearDown(self):
        if self._prev_db_path is None:
            os.environ.pop("DB_PATH", None)
        else:
            os.environ["DB_PATH"] = self._prev_db_path

        import db  # noqa: F401

        importlib.reload(db)
        self._tmpdir.cleanup()

    def _prepare_subject(self):
        self.db.upsert_subject("bpmn", "BPMN", "bpmn", "Prozessmodellierung")
        self.ensure_progress_record("alice", "bpmn")

    def test_advances_after_consistent_success(self):
        self._prepare_subject()
        engine = self.engine_cls(window_size=5, min_attempts=3)

        for idx, score in enumerate([0.9, 0.85, 0.88]):
            engine.process_attempt("alice", "bpmn", f"act-{idx}", score, 1.0)

        progress = self.db.get_user_progress("alice", "bpmn")
        self.assertIsNotNone(progress)
        self.assertEqual(progress["current_level"], SECOND_LEVEL)

        result = engine.evaluate("alice", "bpmn")
        self.assertEqual(result.new_level, SECOND_LEVEL)
        self.assertTrue(result.changed is False)
        self.assertGreaterEqual(result.average_score, 0.85)

    def test_regresses_when_scores_drop(self):
        self._prepare_subject()
        engine = self.engine_cls(window_size=5, min_attempts=3)

        for idx, score in enumerate([0.92, 0.88, 0.9]):
            engine.process_attempt("alice", "bpmn", f"adv-{idx}", score, 1.0)
        # ensure level advanced
        self.assertEqual(
            self.db.get_user_progress("alice", "bpmn")["current_level"], SECOND_LEVEL
        )

        for idx, score in enumerate([0.3, 0.28, 0.25]):
            engine.process_attempt("alice", "bpmn", f"reg-{idx}", score, 1.0)

        progress = self.db.get_user_progress("alice", "bpmn")
        self.assertEqual(progress["current_level"], LOWEST_LEVEL)

    def test_evaluate_with_intervention_system_updates_progress(self):
        self._prepare_subject()
        engine = self.engine_cls(
            window_size=3,
            min_attempts=1,
            intervention_system=LearningInterventionSystem(),
        )

        self.db.record_quiz_attempt(
            "alice",
            "bpmn",
            "int-1",
            0.9,
            1.0,
            pass_threshold=0.8,
        )

        result = engine.evaluate("alice", "bpmn")
        self.assertIsInstance(result.average_score, float)

        progress = self.db.get_user_progress("alice", "bpmn")
        self.assertIsNotNone(progress)
        self.assertGreater(progress["confidence"], 0.0)

    def test_persist_preferences_updates_state(self):
        manager = AdaptiveLearningPathManager()
        user_id = "pref-learner"
        subject_id = "algebra"
        preferences = {
            "modalities": ["Video", "interactive"],
            "pacing": "slow",
            "additional": {"focus_bloom": "K3"},
        }

        manager.persist_preferences(user_id, preferences)
        global_state = self.db.get_learning_path_state(user_id, "__global__")
        self.assertIsNotNone(global_state)
        self.assertIn("preferences", global_state)
        stored_modalities = global_state["preferences"].get("modalities", [])
        self.assertIn("Video", stored_modalities)

        subject_state = manager.get_state(user_id, subject_id)
        self.assertIn("preferences", subject_state)
        subject_modalities = subject_state["preferences"].get("modalities", [])
        self.assertTrue(any(mod.lower() == "video" for mod in subject_modalities))
        self.assertEqual(subject_state["preferences"].get("pacing"), "slow")

    def test_record_quiz_attempt_validates_max_score(self):
        self._prepare_subject()
        with self.assertRaises(ValueError):
            self.db.record_quiz_attempt("bob", "bpmn", "act-1", 1.0, 0.0)

    def test_record_quiz_attempt_rejects_invalid_scores(self):
        self._prepare_subject()
        with self.assertRaises(ValueError):
            self.db.record_quiz_attempt("bob", "bpmn", "act-neg", -0.1, 1.0)
        with self.assertRaises(ValueError):
            self.db.record_quiz_attempt("bob", "bpmn", "act-over", 1.1, 1.0)

    def test_learning_events_and_listing(self):
        self._prepare_subject()
        event_id = self.db.log_learning_event(
            "alice",
            "bpmn",
            "reflection",
            lesson_id="lesson-1",
            score=0.7,
            details={"note": "Needs more practice with gateways"},
        )
        self.assertGreater(event_id, 0)

        events = self.db.list_learning_events(user_id="alice", subject_id="bpmn", limit=5)
        self.assertEqual(len(events), 1)
        stored = events[0]
        self.assertEqual(stored["event_type"], "reflection")
        self.assertIn("note", stored["details"])

    def test_low_confidence_attempts_are_ignored(self):
        self._prepare_subject()
        engine = self.engine_cls(window_size=5, min_attempts=3)

        for idx in range(3):
            self.db.record_quiz_attempt(
                "alice",
                "bpmn",
                f"low-conf-{idx}",
                0.95,
                1.0,
                pass_threshold=0.8,
                confidence=0.2,
            )

        result = engine.evaluate("alice", "bpmn")
        self.assertFalse(result.changed)
        self.assertEqual(result.new_level, LOWEST_LEVEL)
        self.assertAlmostEqual(result.average_score, 0.0)

        bloom_history = self.db.list_bloom_progress_history(user_id="alice", topic="bpmn")
        self.assertEqual(bloom_history, [])

    def test_low_confidence_attempts_do_not_update_td_bkt(self):
        self._prepare_subject()
        prev_flag = os.environ.get("ENABLE_TD_BKT")
        os.environ["ENABLE_TD_BKT"] = "true"
        try:
            engine = self.engine_cls(window_size=3, min_attempts=1)
            initial_theta = 0.45
            self.db.set_theta("alice", "bpmn", initial_theta)

            self.db.record_quiz_attempt(
                "alice",
                "bpmn",
                "td-low",
                0.9,
                1.0,
                pass_threshold=0.8,
                confidence=0.3,
            )

            engine.evaluate("alice", "bpmn")
            theta_after = self.db.get_theta("alice", "bpmn")
            self.assertAlmostEqual(theta_after, initial_theta)
        finally:
            if prev_flag is None:
                os.environ.pop("ENABLE_TD_BKT", None)
            else:
                os.environ["ENABLE_TD_BKT"] = prev_flag

    def test_learning_path_promotion_blocked_by_low_confidence(self):
        self._prepare_subject()
        manager = AdaptiveLearningPathManager()
        sequence = manager._sequence
        if len(sequence) < 2:
            self.skipTest("Bloom sequence requires at least two levels for promotion guard test")

        user_id, subject_id = "alice", "bpmn"
        base_state = {
            "levels": {lvl: 0.0 for lvl in sequence},
            "current_level": sequence[0],
        }
        base_state["levels"][sequence[0]] = 0.7
        self.db.upsert_learning_path_state(user_id, subject_id, base_state)


class MathProgressionStrategyTests(ProgressionEngineTests):
    def setUp(self):
        super().setUp()
        from engines.progression import MathProgressionStrategy

        self.engine_cls = MathProgressionStrategy

    def test_hint_plan_after_repeated_failures(self):
        self.db.upsert_subject("math_algebra", "Mathematik Algebra", "math", "Lineare Gleichungen")
        self.ensure_progress_record("alice", "math_algebra")

        engine = self.engine_cls(window_size=3, min_attempts=1, hint_fail_threshold=2)

        for idx in range(2):
            self.db.record_quiz_attempt(
                "alice",
                "math_algebra",
                f"eq-{idx}",
                0.35,
                1.0,
                pass_threshold=0.7,
                diagnosis="procedural",
            )

        assessment = AssessmentResult(
            user_id="alice",
            domain="math",
            item_id="algebra.linear_equations",
            bloom_level="K3",
            response="Schrittanalyse",
            score=0.35,
            rubric_criteria=[RubricCriterion(id="steps", score=0.35)],
            model_version="gpt-5-codex",
            prompt_version="v1",
            confidence=0.3,
            step_evaluations=[
                AssessmentStepEvaluation(
                    step_id="translate",
                    subskill="algebra.linear_equations.translation",
                    outcome="incorrect",
                    hint="Welche Größen kennst du?",
                    diagnosis="conceptual",
                ),
                AssessmentStepEvaluation(
                    step_id="balance",
                    subskill="algebra.linear_equations.balance",
                    outcome="hint",
                    hint="Addiere -5 auf beiden Seiten.",
                    diagnosis="procedural",
                ),
            ],
            error_patterns=[
                AssessmentErrorPattern(
                    code="sign-flip",
                    description="Vorzeichenwechsel wird ausgelassen",
                    subskill="algebra.linear_equations.balance",
                    occurrences=2,
                )
            ],
        )
        self.db.save_assessment_result(assessment)

        result = engine.evaluate("alice", "math_algebra")
        hint_plan = engine.last_hint_plan

        self.assertEqual(hint_plan.get("preferred_intervention"), "socratic_hint")
        self.assertIn("algebra.linear_equations.balance", hint_plan.get("target_subskill", ""))
        self.assertIn("Socratic", hint_plan.get("message", ""))
        self.assertIn("Visualisierung", hint_plan.get("methods", []))
        self.assertIn("Socratic", result.reason)

    def test_learning_path_transition_capped_to_single_step(self):
        self._prepare_subject()
        manager = AdaptiveLearningPathManager()
        sequence = manager._sequence
        if len(sequence) < 3:
            self.skipTest("Bloom sequence requires at least three levels for transition capping test")

        user_id, subject_id = "alice", "bpmn"

        # Regression attempts should not drop more than one level from the previous state.
        regression_state = {
            "levels": {lvl: 0.5 for lvl in sequence},
            "current_level": sequence[-1],
        }
        regression_state["levels"][sequence[-2]] = 0.5
        self.db.upsert_learning_path_state(user_id, subject_id, regression_state)

        regression_recommendation = manager.update_learning_path(
            user_id=user_id,
            subject_id=subject_id,
            bloom_level=sequence[-2],
            correct=False,
            confidence=0.9,
        )

        self.assertEqual(regression_recommendation.action, "review")
        self.assertEqual(regression_recommendation.recommended_level, sequence[-2])

        state_after_regression = manager.get_state(user_id, subject_id)
        self.assertEqual(state_after_regression["current_level"], sequence[-2])

        # Promotion attempts from a much lower stored level should only advance by one step.
        promotion_state = {
            "levels": {lvl: 0.0 for lvl in sequence},
            "current_level": sequence[0],
        }
        promotion_state["levels"][sequence[-2]] = 0.75
        self.db.upsert_learning_path_state(user_id, subject_id, promotion_state)

        promotion_recommendation = manager.update_learning_path(
            user_id=user_id,
            subject_id=subject_id,
            bloom_level=sequence[-2],
            correct=True,
            confidence=0.95,
        )

        self.assertEqual(promotion_recommendation.action, "promote")
        self.assertEqual(promotion_recommendation.recommended_level, sequence[1])

        state_after_promotion = manager.get_state(user_id, subject_id)
        self.assertEqual(state_after_promotion["current_level"], sequence[1])

    def test_repeated_success_does_not_multi_promote_within_session(self):
        self._prepare_subject()
        manager = AdaptiveLearningPathManager()
        sequence = manager._sequence
        if len(sequence) < 3:
            self.skipTest("Bloom sequence requires at least three levels for session guard test")

        user_id, subject_id = "alice", "bpmn"
        base_state = {
            "levels": {lvl: 0.0 for lvl in sequence},
            "current_level": sequence[0],
        }
        base_state["levels"][sequence[0]] = 0.9
        self.db.upsert_learning_path_state(user_id, subject_id, base_state)

        promotion = manager.update_learning_path(
            user_id=user_id,
            subject_id=subject_id,
            bloom_level=sequence[0],
            correct=True,
            confidence=0.95,
            session_id="session-1",
        )

        self.assertEqual(promotion.action, "promote")
        self.assertEqual(promotion.recommended_level, sequence[1])

        state_after = manager.get_state(user_id, subject_id)
        self.assertIn("last_adjustment", state_after)
        self.assertEqual(state_after["current_level"], sequence[1])

        engine = self.engine_cls(window_size=3, min_attempts=3)
        last_attempt_id = None
        for idx in range(3):
            attempt = self.db.record_quiz_attempt(
                user_id,
                subject_id,
                f"sess1-{idx}",
                0.95,
                1.0,
                pass_threshold=0.8,
                confidence=0.9,
            )
            last_attempt_id = attempt["attempt_id"]

        engine.evaluate(user_id, subject_id, last_attempt_id=int(last_attempt_id))
        bloom_progress = self.db.get_bloom_progress(user_id, subject_id)
        self.assertEqual(bloom_progress["current_level"], sequence[1])

        for idx in range(3, 6):
            attempt = self.db.record_quiz_attempt(
                user_id,
                subject_id,
                f"sess2-{idx}",
                0.97,
                1.0,
                pass_threshold=0.8,
                confidence=0.92,
            )
            last_attempt_id = attempt["attempt_id"]

        engine.evaluate(user_id, subject_id, last_attempt_id=int(last_attempt_id))
        bloom_still = self.db.get_bloom_progress(user_id, subject_id)
        self.assertEqual(bloom_still["current_level"], sequence[1])

        reset_state = manager.get_state(user_id, subject_id)
        reset_state.pop("updated_at", None)
        reset_state.pop("last_adjustment", None)
        self.db.upsert_learning_path_state(user_id, subject_id, reset_state)

        attempt = self.db.record_quiz_attempt(
            user_id,
            subject_id,
            "sess-new",
            0.98,
            1.0,
            pass_threshold=0.8,
            confidence=0.95,
        )

        engine.evaluate(user_id, subject_id, last_attempt_id=int(attempt["attempt_id"]))
        bloom_after = self.db.get_bloom_progress(user_id, subject_id)
        self.assertGreater(sequence.index(bloom_after["current_level"]), 1)

    def test_borderline_attempts_require_additional_evidence(self):
        self._prepare_subject()
        engine = self.engine_cls(window_size=4, min_attempts=3)

        borderline_scores = [0.81, 0.82, 0.83]
        last_attempt_id = None
        for idx, score in enumerate(borderline_scores):
            attempt = self.db.record_quiz_attempt(
                "alice",
                "bpmn",
                f"borderline-pass-{idx}",
                score,
                1.0,
                pass_threshold=0.8,
                confidence=0.55,
            )
            last_attempt_id = attempt["attempt_id"]

        result = engine.evaluate("alice", "bpmn", last_attempt_id=int(last_attempt_id))
        self.assertFalse(result.changed)
        self.assertEqual(result.new_level, LOWEST_LEVEL)
        self.assertEqual(result.reason, "Awaiting more confident evidence before adjusting level.")
        expected_average = round(sum(borderline_scores) / len(borderline_scores), 4)
        self.assertEqual(result.average_score, expected_average)
        self.assertEqual(result.attempts_considered, len(borderline_scores))

        borderline_fail = self.db.record_quiz_attempt(
            "alice",
            "bpmn",
            "borderline-regression",
            0.46,
            1.0,
            pass_threshold=0.8,
            confidence=0.55,
        )

        follow_up = engine.evaluate("alice", "bpmn", last_attempt_id=int(borderline_fail["attempt_id"]))
        self.assertFalse(follow_up.changed)
        self.assertEqual(follow_up.new_level, LOWEST_LEVEL)
        self.assertEqual(
            follow_up.reason,
            "Performance stable — maintaining current level under confidence-weighted review.",
        )
        expected_follow_average = round((sum(borderline_scores) + 0.46) / 4, 4)
        self.assertEqual(follow_up.average_score, expected_follow_average)
        self.assertEqual(follow_up.attempts_considered, len(borderline_scores) + 1)

    def test_evaluation_respects_cooldown_without_new_attempts(self):
        self._prepare_subject()
        engine = self.engine_cls(window_size=3, min_attempts=3)

        last_attempt_id = None
        for idx in range(3):
            attempt = self.db.record_quiz_attempt(
                "alice",
                "bpmn",
                f"cooldown-{idx}",
                0.92,
                1.0,
                pass_threshold=0.8,
                confidence=0.9,
            )
            last_attempt_id = attempt["attempt_id"]

        first_result = engine.evaluate("alice", "bpmn", last_attempt_id=int(last_attempt_id))
        progress_after = self.db.get_user_progress("alice", "bpmn")
        self.assertIsNotNone(progress_after)
        self.assertEqual(progress_after["current_level"], first_result.new_level)

        cooldown_result = engine.evaluate("alice", "bpmn")
        self.assertFalse(cooldown_result.changed)
        self.assertEqual(
            cooldown_result.reason,
            "Keine neuen Versuche seit der letzten Auswertung.",
        )
        self.assertEqual(cooldown_result.new_level, first_result.new_level)
        self.assertEqual(cooldown_result.average_score, first_result.average_score)
        self.assertEqual(
            cooldown_result.attempts_considered,
            first_result.attempts_considered,
        )

    def test_regression_requires_confident_consecutive_evidence(self):
        self._prepare_subject()
        engine = self.engine_cls(window_size=5, min_attempts=3)
        self.db.upsert_user_progress("alice", "bpmn", SECOND_LEVEL, confidence=0.5)

        last_attempt_id = None
        for idx in range(3):
            attempt = self.db.record_quiz_attempt(
                "alice",
                "bpmn",
                f"steady-{idx}",
                0.82,
                1.0,
                pass_threshold=0.8,
                confidence=0.55,
            )
            last_attempt_id = attempt["attempt_id"]

        baseline = engine.evaluate("alice", "bpmn", last_attempt_id=int(last_attempt_id))
        self.assertFalse(baseline.changed)
        self.assertEqual(baseline.new_level, SECOND_LEVEL)
        self.assertEqual(
            baseline.reason,
            "Awaiting more confident evidence before adjusting level.",
        )

        low_conf_fail = self.db.record_quiz_attempt(
            "alice",
            "bpmn",
            "low-confidence-fail",
            0.2,
            1.0,
            pass_threshold=0.8,
            confidence=0.3,
        )
        hold_after_low_conf = engine.evaluate(
            "alice", "bpmn", last_attempt_id=int(low_conf_fail["attempt_id"])
        )
        self.assertFalse(hold_after_low_conf.changed)
        self.assertEqual(hold_after_low_conf.new_level, SECOND_LEVEL)
        self.assertEqual(
            hold_after_low_conf.reason,
            "Awaiting more confident evidence before adjusting level.",
        )

        first_fail = self.db.record_quiz_attempt(
            "alice",
            "bpmn",
            "confident-fail-1",
            0.25,
            1.0,
            pass_threshold=0.8,
            confidence=1.0,
        )
        after_first_fail = engine.evaluate(
            "alice", "bpmn", last_attempt_id=int(first_fail["attempt_id"])
        )
        self.assertFalse(after_first_fail.changed)
        self.assertEqual(after_first_fail.new_level, SECOND_LEVEL)
        self.assertEqual(
            after_first_fail.reason,
            "Performance stable — maintaining current level under confidence-weighted review.",
        )

        second_fail = self.db.record_quiz_attempt(
            "alice",
            "bpmn",
            "confident-fail-2",
            0.2,
            1.0,
            pass_threshold=0.8,
            confidence=1.0,
        )
        regression = engine.evaluate("alice", "bpmn", last_attempt_id=int(second_fail["attempt_id"]))
        self.assertTrue(regression.changed)
        self.assertEqual(regression.new_level, LOWEST_LEVEL)
        self.assertIn("Two-weight struggle detected", regression.reason)
        progress_final = self.db.get_user_progress("alice", "bpmn")
        self.assertEqual(progress_final["current_level"], LOWEST_LEVEL)


if __name__ == "__main__":
    unittest.main()
