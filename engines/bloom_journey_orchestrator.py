"""Bloom's Taxonomy-based Learning Journey Orchestrator.

This module implements a learning journey orchestrator that uses Bloom's Taxonomy
to guide learners through progressive cognitive levels while adapting to their
performance and learning patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from engines.intervention_system import LearningInterventionSystem, LearningPattern
from engines.progression import ProgressionEngine, ProgressionResult
from bloom_levels import BLOOM_LEVELS

logger = logging.getLogger(__name__)

@dataclass
class LearningObjective:
    """Represents a specific learning objective with its Bloom level."""
    id: str
    title: str
    description: str
    bloom_level: str  # K1-K6
    prerequisites: List[str]
    estimated_duration: int  # minutes
    difficulty: float  # 0-1 scale

@dataclass
class LearnerState:
    """Current state of a learner in their journey."""
    user_id: str
    current_objective: str
    completed_objectives: List[str]
    bloom_levels_mastered: Dict[str, float]  # K1-K6 -> mastery score
    learning_preferences: Dict[str, Any]
    last_assessment: Optional[datetime] = None
    intervention_history: List[Dict[str, Any]] = None

class BloomJourneyOrchestrator:
    """Orchestrates learning journeys based on Bloom's Taxonomy."""
    
    def __init__(
        self,
        progression_engine: ProgressionEngine,
        intervention_system: LearningInterventionSystem,
        min_mastery_threshold: float = 0.75,
        max_attempts_before_intervention: int = 3
    ):
        """Initialize the journey orchestrator.
        
        Args:
            progression_engine: Engine for tracking learning progression
            intervention_system: System for providing learning interventions
            min_mastery_threshold: Minimum score needed to consider a level mastered
            max_attempts_before_intervention: Max attempts before triggering intervention
        """
        self.progression_engine = progression_engine
        self.intervention_system = intervention_system
        self.min_mastery_threshold = min_mastery_threshold
        self.max_attempts_before_intervention = max_attempts_before_intervention
        
        # Cache of learning objectives
        self._objectives: Dict[str, LearningObjective] = {}
        self._bloom_sequence = BLOOM_LEVELS.sequence()
        self._learner_cache: Dict[str, LearnerState] = {}
        self._attempt_history: Dict[tuple[str, str], List[float]] = {}
        
    def register_objective(self, objective: LearningObjective) -> None:
        """Register a new learning objective."""
        if objective.bloom_level not in self._bloom_sequence:
            raise ValueError(f"Invalid Bloom level: {objective.bloom_level}")
            
        # Validate prerequisites exist
        for prereq in objective.prerequisites:
            if prereq not in self._objectives and prereq != "none":
                raise ValueError(f"Prerequisite objective not found: {prereq}")
                
        self._objectives[objective.id] = objective
        
    def get_next_objective(
        self,
        learner: LearnerState,
        subject_id: str
    ) -> Optional[LearningObjective]:
        """Determine the next most appropriate learning objective."""
        
        # Get current mastery levels
        current_mastery = self._get_current_mastery(learner, subject_id)
        
        # Find objectives at appropriate Bloom level
        current_bloom_level = self._determine_current_bloom_level(current_mastery)
        candidate_objectives = [
            obj for obj in self._objectives.values()
            if obj.bloom_level == current_bloom_level
            and obj.id not in learner.completed_objectives
            and all(p in learner.completed_objectives or p == "none" 
                   for p in obj.prerequisites)
        ]
        
        if not candidate_objectives:
            remaining = [
                obj
                for obj in self._objectives.values()
                if obj.id not in learner.completed_objectives
                and all(p in learner.completed_objectives or p == "none" for p in obj.prerequisites)
            ]
            if not remaining:
                return None
            remaining.sort(key=lambda obj: self._bloom_sequence.index(obj.bloom_level))
            return remaining[0]

        # Select best matching objective based on learner state
        return self._select_best_objective(candidate_objectives, learner)
        
    def assess_progress(
        self,
        learner: LearnerState,
        objective_id: str,
        assessment_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess learner progress and determine next steps."""
        
        objective = self._objectives.get(objective_id)
        if not objective:
            raise ValueError(f"Unknown objective: {objective_id}")
            
        # Update progression tracking
        normalized_score = float(assessment_result.get("normalized_score", 0.0))
        subject_id = assessment_result.get("subject_id") or objective.bloom_level
        self.progression_engine.record_attempt(
            user_id=learner.user_id,
            subject_id=subject_id,
            activity_id=objective_id,
            score=normalized_score,
            max_score=1.0,
            pass_threshold=self.min_mastery_threshold
        )

        history_key = (learner.user_id, objective_id)
        history = self._attempt_history.setdefault(history_key, [])
        history.append(normalized_score)
        if len(history) > self.max_attempts_before_intervention:
            self._attempt_history[history_key] = history[-self.max_attempts_before_intervention:]

        # Check if intervention is needed
        if self._should_intervene(learner, objective, assessment_result):
            pattern = self._create_learning_pattern(learner, assessment_result)
            intervention = self.intervention_system.monitor_progress(
                learner.user_id, pattern
            )
            if intervention:
                return {
                    "status": "intervention_needed",
                    "intervention": asdict(intervention),
                    "next_objective": objective_id,
                }
            return {
                "status": "intervention_needed",
                "intervention": None,
                "next_objective": objective_id,
            }
        
        # Check if objective is mastered
        current_mastery = learner.bloom_levels_mastered.get(objective.bloom_level, 0.0)
        learner.bloom_levels_mastered[objective.bloom_level] = max(current_mastery, normalized_score)

        if normalized_score >= self.min_mastery_threshold:
            if objective_id not in learner.completed_objectives:
                learner.completed_objectives.append(objective_id)
            self._learner_cache[learner.user_id] = learner
            return {
                "status": "objective_completed",
                "mastery_level": normalized_score,
                "next_objective": self.get_next_objective(learner, objective.bloom_level)
            }

        self._learner_cache[learner.user_id] = learner
        return {
            "status": "in_progress",
            "mastery_level": normalized_score,
            "next_objective": objective_id
        }

    def _get_current_mastery(
        self,
        learner: LearnerState | str,
        subject_id: str
    ) -> Dict[str, float]:
        """Get current mastery levels for each Bloom level."""

        learner_state: Optional[LearnerState]
        if isinstance(learner, LearnerState):
            learner_state = learner
        else:
            learner_state = self._learner_cache.get(learner)

        mastery: Dict[str, float] = {}
        fallback_progress: ProgressionResult | None = None

        for level in self._bloom_sequence:
            score = None
            if learner_state and learner_state.bloom_levels_mastered:
                score = learner_state.bloom_levels_mastered.get(level)

            if score is not None:
                mastery[level] = float(score or 0.0)
                continue

            if fallback_progress is None:
                fallback_progress = self.progression_engine.evaluate(
                    learner_state.user_id if learner_state else str(learner),
                    subject_id
                )
            mastery[level] = float(fallback_progress.average_score)

        return mastery
        
    def _determine_current_bloom_level(
        self,
        mastery: Dict[str, float]
    ) -> str:
        """Determine appropriate Bloom level based on mastery."""
        for level in self._bloom_sequence:
            if mastery.get(level, 0.0) < self.min_mastery_threshold:
                return level
        return self._bloom_sequence[-1]  # Highest level if all mastered
        
    def _select_best_objective(
        self,
        candidates: List[LearningObjective],
        learner: LearnerState
    ) -> LearningObjective:
        """Select the most appropriate objective from candidates."""
        # TODO: Implement more sophisticated selection based on:
        # - Learning preferences
        # - Previous performance patterns
        # - Time constraints
        # - Difficulty progression
        return candidates[0]  # Simple selection for now
        
    def _should_intervene(
        self,
        learner: LearnerState,
        objective: LearningObjective,
        assessment: Dict[str, Any]
    ) -> bool:
        """Determine if intervention is needed."""
        subject_id = assessment.get("subject_id") or objective.bloom_level
        history_key = (learner.user_id, objective.id)
        recent_scores = self._attempt_history.get(history_key, [])
        if len(recent_scores) < self.max_attempts_before_intervention:
            return False

        window = recent_scores[-self.max_attempts_before_intervention:]
        avg_score = sum(window) / len(window)
        return avg_score < self.min_mastery_threshold
        
    def _create_learning_pattern(
        self,
        learner: LearnerState,
        assessment: Dict[str, Any]
    ) -> LearningPattern:
        """Create learning pattern from recent activity."""
        subject_id = assessment.get("subject_id") or learner.current_objective
        recent_attempts = self.progression_engine.get_recent_attempts(
            learner.user_id,
            subject_id,
            limit=5
        )

        return LearningPattern(
            response_times=[a.response_time for a in recent_attempts if a.response_time],
            accuracy_scores=[a.normalized_score for a in recent_attempts],
            engagement_levels=[a.engagement_level for a in recent_attempts if a.engagement_level],
            hint_usage=[a.hint_usage for a in recent_attempts if a.hint_usage],
            timestamps=[datetime.fromisoformat(a.created_at) for a in recent_attempts]
        )