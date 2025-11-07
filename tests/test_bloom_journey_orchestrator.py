"""Tests for the Bloom's Taxonomy-based Learning Journey Orchestrator."""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from engines.bloom_journey_orchestrator import (
    BloomJourneyOrchestrator,
    LearningObjective,
    LearnerState
)
from engines.intervention_system import LearningInterventionSystem
from engines.progression import ProgressionEngine

@pytest.fixture
def progression_engine():
    return ProgressionEngine()

@pytest.fixture
def intervention_system():
    return LearningInterventionSystem()

@pytest.fixture
def orchestrator(progression_engine, intervention_system):
    return BloomJourneyOrchestrator(
        progression_engine=progression_engine,
        intervention_system=intervention_system
    )

@pytest.fixture
def sample_objectives():
    return [
        LearningObjective(
            id="obj1",
            title="Basic Concepts",
            description="Understanding basic terminology",
            bloom_level="K1",
            prerequisites=["none"],
            estimated_duration=30,
            difficulty=0.3
        ),
        LearningObjective(
            id="obj2",
            title="Apply Concepts",
            description="Apply concepts to simple problems",
            bloom_level="K3",
            prerequisites=["obj1"],
            estimated_duration=45,
            difficulty=0.5
        ),
        LearningObjective(
            id="obj3",
            title="Analyze Problems",
            description="Break down complex problems",
            bloom_level="K4",
            prerequisites=["obj2"],
            estimated_duration=60,
            difficulty=0.7
        )
    ]

@pytest.fixture
def sample_learner():
    return LearnerState(
        user_id="test_user",
        current_objective="obj1",
        completed_objectives=[],
        bloom_levels_mastered={
            "K1": 0.0,
            "K2": 0.0,
            "K3": 0.0,
            "K4": 0.0,
            "K5": 0.0,
            "K6": 0.0
        },
        learning_preferences={},
        intervention_history=[]
    )

def test_register_objective(orchestrator, sample_objectives):
    """Test registering learning objectives."""
    for obj in sample_objectives:
        orchestrator.register_objective(obj)
        
    with pytest.raises(ValueError):
        # Test invalid Bloom level
        invalid_obj = LearningObjective(
            id="invalid",
            title="Invalid",
            description="Invalid",
            bloom_level="K7",  # Invalid level
            prerequisites=["none"],
            estimated_duration=30,
            difficulty=0.3
        )
        orchestrator.register_objective(invalid_obj)
        
    with pytest.raises(ValueError):
        # Test invalid prerequisite
        invalid_prereq = LearningObjective(
            id="invalid_prereq",
            title="Invalid Prereq",
            description="Invalid prerequisite",
            bloom_level="K1",
            prerequisites=["nonexistent"],
            estimated_duration=30,
            difficulty=0.3
        )
        orchestrator.register_objective(invalid_prereq)

def test_get_next_objective(orchestrator, sample_objectives, sample_learner):
    """Test getting next appropriate objective."""
    # Register objectives
    for obj in sample_objectives:
        orchestrator.register_objective(obj)
        
    # First objective should be obj1 (K1 level)
    next_obj = orchestrator.get_next_objective(sample_learner, "test_subject")
    assert next_obj.id == "obj1"
    
    # Complete obj1
    sample_learner.completed_objectives.append("obj1")
    sample_learner.bloom_levels_mastered["K1"] = 0.8
    
    # Next should be obj2
    next_obj = orchestrator.get_next_objective(sample_learner, "test_subject")
    assert next_obj.id == "obj2"

def test_assess_progress(orchestrator, sample_objectives, sample_learner):
    """Test progress assessment and intervention triggering."""
    # Register objectives
    for obj in sample_objectives:
        orchestrator.register_objective(obj)
        
    # Test successful completion
    result = orchestrator.assess_progress(
        sample_learner,
        "obj1",
        {
            "normalized_score": 0.8,
            "subject_id": "test_subject"
        }
    )
    assert result["status"] == "objective_completed"
    assert "obj1" in sample_learner.completed_objectives
    
    # Test intervention needed
    low_scores = [
        {"normalized_score": 0.3, "subject_id": "test_subject"}
        for _ in range(3)
    ]
    for assessment in low_scores:
        result = orchestrator.assess_progress(
            sample_learner,
            "obj2",
            assessment
        )
    assert result["status"] == "intervention_needed"
    
def test_mastery_tracking(orchestrator, sample_objectives, sample_learner):
    """Test tracking of mastery levels across Bloom's taxonomy."""
    # Register objectives
    for obj in sample_objectives:
        orchestrator.register_objective(obj)
    
    # Test progression through levels
    assessments = [
        ("obj1", 0.8),  # Master K1
        ("obj2", 0.3),  # Struggle with K3
        ("obj2", 0.85), # Master K3
        ("obj3", 0.4),  # Initial attempt at K4
    ]
    
    for obj_id, score in assessments:
        result = orchestrator.assess_progress(
            sample_learner,
            obj_id,
            {
                "normalized_score": score,
                "subject_id": "test_subject"
            }
        )
        
        if score >= orchestrator.min_mastery_threshold:
            assert obj_id in sample_learner.completed_objectives
            
    # Verify mastery tracking
    mastery = orchestrator._get_current_mastery(
        sample_learner.user_id,
        "test_subject"
    )
    assert mastery["K1"] >= 0.8  # Should be mastered
    assert mastery["K3"] >= 0.8  # Should be mastered
    assert mastery["K4"] < 0.8   # Should not be mastered

def test_intervention_timing(orchestrator, sample_objectives, sample_learner):
    """Test that interventions are triggered at appropriate times."""
    # Register objectives
    for obj in sample_objectives:
        orchestrator.register_objective(obj)
        
    # Should not trigger intervention on first attempt
    result = orchestrator.assess_progress(
        sample_learner,
        "obj1",
        {
            "normalized_score": 0.3,
            "subject_id": "test_subject"
        }
    )
    assert result["status"] == "in_progress"
    
    # Should trigger intervention after max_attempts_before_intervention
    for _ in range(orchestrator.max_attempts_before_intervention):
        result = orchestrator.assess_progress(
            sample_learner,
            "obj1",
            {
                "normalized_score": 0.3,
                "subject_id": "test_subject"
            }
        )
    assert result["status"] == "intervention_needed"