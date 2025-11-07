import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

import db
from schemas import AssessmentResult, RubricCriterion


def make_assessment(user_id: str, *, confidence: float, score: float = 0.5) -> AssessmentResult:
    return AssessmentResult(
        user_id=user_id,
        domain="math",
        item_id="item-1",
        bloom_level="K2",
        response="answer",
        score=score,
        rubric_criteria=[RubricCriterion(id="accuracy", score=score)],
        model_version="test",
        prompt_version="chat.v1",
        confidence=confidence,
        source="direct",
        latency_ms=1200,
        created_at=datetime.now(timezone.utc),
    )


def test_compute_teacher_analytics_flags_stuck_cohort(temp_db):
    db.upsert_learning_path_state(
        "learner",
        "math",
        {
            "current_level": "K2",
            "levels": {"K1": 0.8, "K2": 0.35},
            "history": [
                {"bloom_level": "K2", "delta": -0.08, "correct": False, "confidence": 0.32},
                {"bloom_level": "K2", "delta": -0.04, "correct": False, "confidence": 0.28},
                {"bloom_level": "K2", "delta": -0.02, "correct": False, "confidence": 0.31},
                {"bloom_level": "K2", "delta": 0.0, "correct": False, "confidence": 0.29},
            ],
        },
    )
    db.log_journey_update(
        "learner",
        "hint_requested",
        {"subject_id": "math", "details": {"type": "hint_request"}},
    )
    db.save_assessment_result(make_assessment("learner", confidence=0.2, score=0.42))

    snapshot = db.compute_teacher_analytics(window_days=30)
    entry = next(item for item in snapshot if item["user_id"] == "learner" and item["subject_id"] == "math")

    assert entry["hint_count"] == 1
    assert entry["low_confidence_count"] >= 1
    assert entry["stuck_flag"] is True
    assert entry["flag_low_confidence"] is True
    assert entry["flag_regression"] is True
    assert entry["flag_high_hints"] is False
    assert "low_confidence" in entry["flagged_reasons"]
    assert "regression" in entry["flagged_reasons"]
    assert entry["history_tail"]
    assert entry["current_level"] == "K2"


def test_apply_learning_path_override_persists_state_and_metrics(temp_db):
    db.upsert_learning_path_state(
        "mentor",
        "math",
        {"current_level": "K2", "levels": {"K2": 0.4}},
    )

    state = db.apply_learning_path_override(
        "mentor",
        "math",
        target_level="K3",
        notes="Advance to project work",
        applied_by="teacher@example.com",
    )

    assert state["current_level"] == "K3"
    assert state["manual_override"]["target_level"] == "K3"
    assert state["manual_override"]["notes"] == "Advance to project work"

    snapshot = db.compute_teacher_analytics(window_days=7)
    entry = next(item for item in snapshot if item["user_id"] == "mentor" and item["subject_id"] == "math")
    assert entry["manual_override"]["target_level"] == "K3"
    assert entry["current_level"] == "K3"
    assert entry["manual_override"]["applied_by"] == "teacher@example.com"


def test_teacher_analytics_merges_normalized_gain_pairs(temp_db):
    import app as app_module

    db.record_pretest_attempt(
        learner_id="gain-learner",
        topic="math",
        score=0.4,
        max_score=1.0,
        instrument={"title": "Pre math", "description": "", "items": [], "metadata": {}},
    )
    db.record_posttest_attempt(
        learner_id="gain-learner",
        topic="math",
        score=0.7,
        max_score=1.0,
        instrument={"title": "Post math", "description": "", "items": [], "metadata": {}},
    )

    results = app_module.teacher_analytics()
    entry = next(item for item in results if item["user_id"] == "gain-learner" and item["topic"] == "math")

    assert entry["pre_score"] == pytest.approx(0.4)
    assert entry["post_score"] == pytest.approx(0.7)
    assert entry["score_delta"] == pytest.approx(0.3, abs=1e-6)
    assert entry["normalized_gain"] == pytest.approx(0.5)


def test_teacher_analytics_flags_negative_gain_as_regression(temp_db):
    import app as app_module

    db.record_pretest_attempt(
        learner_id="regress-learner",
        topic="math",
        score=0.8,
        max_score=1.0,
        instrument={"title": "Pre math", "description": "", "items": [], "metadata": {}},
    )
    db.record_posttest_attempt(
        learner_id="regress-learner",
        topic="math",
        score=0.5,
        max_score=1.0,
        instrument={"title": "Post math", "description": "", "items": [], "metadata": {}},
    )

    results = app_module.teacher_analytics()
    entry = next(item for item in results if item["user_id"] == "regress-learner" and item["topic"] == "math")

    assert entry["flag_regression"] is True
    assert "regression" in entry["flagged_reasons"]
    assert entry["stuck_flag"] is True


def test_teacher_analytics_confidence_interval_from_assessments(temp_db):
    learner_id = "ci-learner"
    db.upsert_learning_path_state(
        learner_id,
        "math",
        {"current_level": "K2", "levels": {"K2": 0.4}},
    )
    db.save_assessment_result(make_assessment(learner_id, confidence=0.3, score=0.4))
    db.save_assessment_result(make_assessment(learner_id, confidence=0.5, score=0.5))
    db.save_assessment_result(make_assessment(learner_id, confidence=0.8, score=0.6))

    snapshot = db.compute_teacher_analytics(window_days=30)
    entry = next(item for item in snapshot if item["user_id"] == learner_id and item["subject_id"] == "math")

    assert entry["confidence_interval_lower"] is not None
    assert entry["confidence_interval_upper"] is not None
    assert entry["confidence_interval_margin"] is not None
    assert entry["confidence_interval_width"] is not None
    assert entry["confidence_interval_width"] > 0
    assert entry["confidence_interval_confidence_level"] == pytest.approx(0.95)
    assert entry["confidence_interval_sample_size"] == 3


def test_teacher_analytics_includes_timelines_and_feedback(temp_db):
    import app as app_module

    learner_id = "timeline-learner"
    subject_id = "math"
    db.ensure_user(learner_id)
    db.upsert_learning_path_state(
        learner_id,
        subject_id,
        {"current_level": "K2", "history": []},
    )
    db.log_learning_path_event(
        learner_id,
        subject_id,
        bloom_level="K1",
        action="initial_recommendation",
        reason_code="diagnostic",
        reason="Diagnostic placement",
        confidence=0.75,
        evidence={"source": "assessment", "score": 0.42},
    )
    db.log_learning_path_event(
        learner_id,
        subject_id,
        bloom_level="K2",
        action="advance",
        reason_code="mastery",
        reason="Teacher adjustment",
        confidence=0.88,
        evidence={"source": "teacher", "notes": "Reviewed portfolio"},
    )
    db.upsert_bloom_progress(learner_id, subject_id, "K1", reason="Initial state", average_score=0.4)
    db.upsert_bloom_progress(learner_id, subject_id, "K2", reason="Growth", average_score=0.8)
    db.store_feedback(
        learner_id,
        answer_id="ans-1",
        rating="up",
        comment="Great explanation",
        confidence=0.9,
    )
    db.store_feedback(
        learner_id,
        answer_id="ans-2",
        rating="down",
        comment="Needs more detail",
        confidence=0.4,
    )

    results = app_module.teacher_analytics(window_days=30)
    entry = next(item for item in results if item["user_id"] == learner_id and item["topic"] == subject_id)

    timeline = entry["learning_path_events"]
    assert isinstance(timeline, list)
    assert len(timeline) >= 2
    created_order = [event.get("created_at") for event in timeline if event.get("created_at")]
    assert created_order == sorted(created_order)

    latest = entry["latest_path_event"]
    assert latest is not None
    assert latest["reason"] == "Teacher adjustment"
    assert entry["latest_path_rationale"] == "Teacher adjustment"
    assert entry["latest_path_evidence_summary"]
    assert "source=teacher" in entry["latest_path_evidence_summary"]

    bloom_history = entry["bloom_history"]
    assert bloom_history
    bloom_order = [item.get("created_at") for item in bloom_history if item.get("created_at")]
    assert bloom_order == sorted(bloom_order)

    feedback_summary = entry["feedback_summary"]
    assert feedback_summary["total"] == 2
    assert feedback_summary["ratings"]["up"] == 1
    assert feedback_summary["ratings"]["down"] == 1


def test_profile_includes_feedback_summary(temp_db):
    import app as app_module

    learner_id = "profile-feedback"
    db.ensure_user(learner_id)
    db.store_feedback(learner_id, answer_id="profile-ans-1", rating="up", comment="Helpful", confidence=0.8)
    db.store_feedback(learner_id, answer_id="profile-ans-2", rating="flag", comment="Check", confidence=0.5)

    payload = app_module.profile(learner_id)
    summary = payload.get("feedback_summary")
    assert summary is not None
    assert summary["total"] == 2
    assert summary["ratings"]["up"] == 1
    assert summary["ratings"]["flag"] == 1
