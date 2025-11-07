"""Test cases for the learning intervention system."""

import pytest
from datetime import datetime, timedelta

from engines.intervention_system import (
    LearningInterventionSystem,
    LearningPattern,
    InterventionTrigger
)

def create_test_pattern(
    response_times=None,
    accuracy_scores=None,
    engagement_levels=None,
    hint_usage=None,
    time_delta_minutes=5,
    *,
    objective_id=None,
    bloom_level=None,
    progress_snapshot=None,
) -> LearningPattern:
    """Create a test learning pattern with default values."""
    now = datetime.utcnow()
    timestamps = [
        now - timedelta(minutes=i * time_delta_minutes)
        for i in range(len(accuracy_scores or []))
    ]
    
    return LearningPattern(
        response_times=response_times or [],
        accuracy_scores=accuracy_scores or [],
        engagement_levels=engagement_levels or [],
        hint_usage=hint_usage or [],
        timestamps=timestamps,
        objective_id=objective_id,
        bloom_level=bloom_level,
        progress_snapshot=progress_snapshot,
    )

def test_struggle_detection():
    """Test detection of learning struggle patterns."""
    system = LearningInterventionSystem()
    system.min_data_points = 3
    
    # Test pattern showing consistent low scores
    pattern = create_test_pattern(
        accuracy_scores=[0.3, 0.2, 0.35],
        response_times=[60, 70, 65],
        engagement_levels=[0.7, 0.6, 0.65],
        hint_usage=[0.4, 0.5, 0.45]
    )
    
    trigger = system.monitor_progress("test_user", pattern)
    assert trigger is not None
    assert trigger.type == "struggle"
    assert trigger.confidence > 0.6

def test_boredom_detection():
    """Test detection of student boredom patterns."""
    system = LearningInterventionSystem()
    system.min_data_points = 3
    
    # Test pattern showing declining engagement
    pattern = create_test_pattern(
        accuracy_scores=[0.9, 0.95, 0.9],
        response_times=[20, 15, 10],
        engagement_levels=[0.5, 0.4, 0.3],
        hint_usage=[0.1, 0.1, 0.1]
    )
    
    trigger = system.monitor_progress("test_user", pattern)
    assert trigger is not None
    assert trigger.type == "boredom"
    assert trigger.confidence > 0.6

def test_fatigue_detection():
    """Test detection of learning fatigue patterns."""
    system = LearningInterventionSystem()
    system.min_data_points = 3
    
    # Test pattern showing increasing response times
    pattern = create_test_pattern(
        accuracy_scores=[0.8, 0.75, 0.7],
        response_times=[130, 140, 150],
        engagement_levels=[0.7, 0.6, 0.5],
        hint_usage=[0.2, 0.3, 0.4]
    )
    
    trigger = system.monitor_progress("test_user", pattern)
    assert trigger is not None
    assert trigger.type == "fatigue"
    assert trigger.confidence > 0.6

def test_confusion_detection():
    """Test detection of student confusion patterns."""
    system = LearningInterventionSystem()
    system.min_data_points = 3
    
    # Test pattern showing increasing hint usage
    pattern = create_test_pattern(
        accuracy_scores=[0.6, 0.5, 0.4],
        response_times=[45, 60, 75],
        engagement_levels=[0.8, 0.7, 0.6],
        hint_usage=[0.6, 0.7, 0.8]
    )
    
    trigger = system.monitor_progress("test_user", pattern)
    assert trigger is not None
    assert trigger.type == "confusion"
    assert trigger.confidence > 0.6

def test_intervention_generation():
    """Test generation of appropriate interventions."""
    system = LearningInterventionSystem()
    
    # Test struggle intervention
    trigger = InterventionTrigger(
        type="struggle",
        confidence=0.8,
        detected_at=datetime.utcnow(),
        context={
            "recent_accuracy": [0.3, 0.2, 0.35],
            "hint_usage": [0.4, 0.5, 0.45],
            "avg_response_time": 65.0
        }
    )
    
    intervention = system.generate_intervention(
        trigger,
        user_profile={"preferred_learning_style": "visual"}
    )
    assert intervention["type"] == "struggle_support"
    assert len(intervention["suggestions"]) >= 2

def test_intervention_cooldown():
    """Test that interventions respect cooldown period."""
    system = LearningInterventionSystem()
    system.min_data_points = 3
    
    # Create a pattern that would trigger intervention
    pattern = create_test_pattern(
        accuracy_scores=[0.3, 0.2, 0.35],
        response_times=[60, 70, 65],
        engagement_levels=[0.7, 0.6, 0.65],
        hint_usage=[0.4, 0.5, 0.45]
    )
    
    # First intervention should trigger
    trigger1 = system.monitor_progress("test_user", pattern)
    assert trigger1 is not None
    
    # Immediate second attempt should not trigger due to cooldown
    trigger2 = system.monitor_progress("test_user", pattern)
    assert trigger2 is None

def test_multiple_triggers():
    """Test handling of multiple simultaneous triggers."""
    system = LearningInterventionSystem()
    system.min_data_points = 3
    
    # Create a pattern that could trigger multiple interventions
    pattern = create_test_pattern(
        accuracy_scores=[0.3, 0.2, 0.35],  # Low scores (struggle)
        response_times=[130, 140, 150],     # High response times (fatigue)
        engagement_levels=[0.7, 0.6, 0.65],
        hint_usage=[0.4, 0.5, 0.45]
    )
    
    trigger = system.monitor_progress("test_user", pattern)
    assert trigger is not None
    # Should select the trigger with highest confidence
    assert trigger.type in ["struggle", "fatigue"]

def test_edge_cases():
    """Test system behavior with edge cases."""
    system = LearningInterventionSystem()
    system.min_data_points = 3
    
    # Test with empty pattern
    empty_pattern = create_test_pattern(
        accuracy_scores=[],
        response_times=[],
        engagement_levels=[],
        hint_usage=[]
    )
    trigger = system.monitor_progress("test_user", empty_pattern)
    assert trigger is None
    
    # Test with single data point
    single_pattern = create_test_pattern(
        accuracy_scores=[0.3],
        response_times=[60],
        engagement_levels=[0.7],
        hint_usage=[0.4]
    )
    trigger = system.monitor_progress("test_user", single_pattern)
    assert trigger is None  # Should require multiple data points
    
    # Test with invalid values
    invalid_pattern = create_test_pattern(
        accuracy_scores=[-0.1, 1.5, 0.5],  # Invalid scores
        response_times=[60, 70, 65],
        engagement_levels=[0.7, 0.6, 0.65],
        hint_usage=[0.4, 0.5, 0.45]
    )
    with pytest.raises(ValueError):
        system.monitor_progress("test_user", invalid_pattern)


def test_trigger_includes_objective_context():
    """Triggers should carry Bloom/objective context for downstream tailoring."""
    system = LearningInterventionSystem()
    system.min_data_points = 3

    progress_snapshot = {
        "subject_id": "math",
        "recommended_objective_label": "Quadratic Functions",
    }

    pattern = create_test_pattern(
        accuracy_scores=[0.3, 0.25, 0.2],
        response_times=[70, 80, 90],
        engagement_levels=[0.6, 0.55, 0.5],
        hint_usage=[0.4, 0.45, 0.5],
        objective_id="skill.quadratics",
        bloom_level="K3",
        progress_snapshot=progress_snapshot,
    )

    trigger = system.monitor_progress("test_user", pattern)
    assert trigger is not None
    assert trigger.type == "struggle"
    assert trigger.context.get("objective_id") == "skill.quadratics"
    assert trigger.context.get("bloom_level") == "K3"
    snapshot = trigger.context.get("progress_snapshot")
    assert isinstance(snapshot, dict)
    assert snapshot.get("recommended_objective_label") == "Quadratic Functions"
