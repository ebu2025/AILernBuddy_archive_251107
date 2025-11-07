import db
from learning_path import AdaptiveLearningPathManager
from schemas import AssessmentResult, RubricCriterion


def make_assessment(user_id: str, score: float = 0.9, level: str = "K2") -> AssessmentResult:
    return AssessmentResult(
        user_id=user_id,
        domain="math",
        item_id="item-1",
        bloom_level=level,
        response="Answer",
        score=score,
        rubric_criteria=[RubricCriterion(id="accuracy", score=score)],
        model_version="test",
        prompt_version="chat.v1",
        confidence=0.85,
        source="direct",
        latency_ms=4200,
    )


def test_learning_path_updates_state(temp_db):
    manager = AdaptiveLearningPathManager()
    assessment = make_assessment("learner")

    recommendation = manager.update_from_assessment(assessment)
    assert recommendation is not None
    assert recommendation.recommended_level in manager._sequence
    assert recommendation.reason_code
    assert recommendation.evidence
    assert recommendation.evidence.get("items")

    state = db.get_learning_path_state("learner", "math")
    assert state is not None
    assert "levels" in state
    assert state["current_level"] == recommendation.recommended_level

    events = db.list_learning_path_events("learner", "math", limit=5)
    assert events
    assert events[0]["action"] == recommendation.action
    assert events[0]["reason_code"] == recommendation.reason_code
    evidence = events[0].get("evidence") or {}
    items = evidence.get("items") if isinstance(evidence, dict) else None
    if items:
        assert items[0].get("id") == assessment.item_id


def test_learning_path_regression(temp_db):
    manager = AdaptiveLearningPathManager()
    assessment = make_assessment("learner", score=0.2, level="K3")

    recommendation = manager.update_from_assessment(assessment)
    assert recommendation is not None
    assert recommendation.action in {"review", "stabilise"}

    summary = db.aggregate_feedback(answer_id="noop")
    assert summary["total"] == 0
