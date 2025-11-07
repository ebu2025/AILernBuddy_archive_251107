"""Real-time learning intervention system for proactive support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import math

@dataclass
class LearningPattern:
    response_times: List[float]
    accuracy_scores: List[float]
    engagement_levels: List[float]
    hint_usage: List[float]
    timestamps: List[datetime]
    objective_id: Optional[str] = None
    bloom_level: Optional[str] = None
    progress_snapshot: Optional[Dict[str, Any]] = None

@dataclass
class InterventionTrigger:
    type: str  # 'struggle', 'boredom', 'fatigue', 'confusion'
    confidence: float
    detected_at: datetime
    context: Dict[str, Any]

class LearningInterventionSystem:
    def __init__(self):
        self.engagement_threshold = 0.6
        self.struggle_threshold = 0.4
        self.confusion_threshold = 0.5
        self.response_time_threshold = 120  # seconds
        
        self.intervention_cooldown = timedelta(minutes=5)
        self.pattern_window = timedelta(minutes=15)
        self.min_data_points = 5  # Minimum data points for reliable detection
        
        self.last_intervention: Dict[str, datetime] = {}  # Per-user cooldown
        self.max_interventions_per_hour = 5  # Rate limiting

    def monitor_progress(self,
                        user_id: str,
                        current_pattern: LearningPattern) -> Optional[InterventionTrigger]:
        """Monitor learning patterns and detect need for intervention."""
        now = datetime.utcnow()
        
        # Validate input data
        self._validate_pattern(current_pattern)
            
        # Check if we have enough data points
        if len(current_pattern.accuracy_scores) < self.min_data_points:
            return None
            
        # Check cooldown per user
        if (user_id in self.last_intervention and 
            now - self.last_intervention[user_id] < self.intervention_cooldown):
            return None
            
        # Rate limiting check
        recent_interventions = sum(
            1 for timestamp in self.last_intervention.values()
            if now - timestamp < timedelta(hours=1)
        )
        if recent_interventions >= self.max_interventions_per_hour:
            return None

        # Analyze patterns
        triggers = []
        
        base_context = self._base_context(current_pattern)

        # Check for struggle
        if self._detect_struggle(current_pattern):
            triggers.append(
                InterventionTrigger(
                    type="struggle",
                    confidence=self._calculate_struggle_confidence(current_pattern),
                    detected_at=datetime.utcnow(),
                    context=self._get_struggle_context(current_pattern, base_context)
                )
            )

        # Check for boredom
        if self._detect_boredom(current_pattern):
            triggers.append(
                InterventionTrigger(
                    type="boredom",
                    confidence=self._calculate_boredom_confidence(current_pattern),
                    detected_at=datetime.utcnow(),
                    context=self._get_boredom_context(current_pattern, base_context)
                )
            )

        # Check for fatigue
        if self._detect_fatigue(current_pattern):
            triggers.append(
                InterventionTrigger(
                    type="fatigue",
                    confidence=self._calculate_fatigue_confidence(current_pattern),
                    detected_at=datetime.utcnow(),
                    context=self._get_fatigue_context(current_pattern, base_context)
                )
            )

        # Check for confusion
        if self._detect_confusion(current_pattern):
            triggers.append(
                InterventionTrigger(
                    type="confusion",
                    confidence=self._calculate_confusion_confidence(current_pattern),
                    detected_at=datetime.utcnow(),
                    context=self._get_confusion_context(current_pattern, base_context)
                )
            )

        # Select highest confidence trigger
        if triggers:
            selected_trigger = max(triggers, key=lambda t: t.confidence)
            self.last_intervention[user_id] = datetime.utcnow()
            
            # Clean up old intervention records
            self._cleanup_old_interventions()
            return selected_trigger

        return None

    def _validate_pattern(self, pattern: LearningPattern) -> None:
        """Validate learning pattern data."""
        # Validate scores are in [0, 1] range
        if not all(0 <= score <= 1 for score in pattern.accuracy_scores):
            raise ValueError("Accuracy scores must be between 0 and 1")

        # Validate engagement levels
        if not all(0 <= level <= 1 for level in pattern.engagement_levels):
            raise ValueError("Engagement levels must be between 0 and 1")

        # Validate hint usage
        if not all(0 <= usage <= 1 for usage in pattern.hint_usage):
            raise ValueError("Hint usage values must be between 0 and 1")

        # Validate response times are positive
        if not all(time >= 0 for time in pattern.response_times):
            raise ValueError("Response times must be non-negative")

        # Validate timestamps are not in the future
        now = datetime.utcnow()
        if any(ts > now for ts in pattern.timestamps):
            raise ValueError("Timestamps cannot be in the future")

        # Validate arrays have same length if not empty
        lengths = {len(arr) for arr in [
            pattern.response_times,
            pattern.accuracy_scores,
            pattern.engagement_levels,
            pattern.hint_usage,
            pattern.timestamps
        ] if arr}

        if len(lengths) > 1:
            raise ValueError("Learning pattern series must have matching lengths")

        if pattern.progress_snapshot is not None and not isinstance(pattern.progress_snapshot, dict):
            raise ValueError("progress_snapshot must be a mapping when provided")
            
    def _cleanup_old_interventions(self) -> None:
        """Clean up intervention records older than 1 hour."""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=1)
        
        self.last_intervention = {
            user_id: timestamp
            for user_id, timestamp in self.last_intervention.items()
            if timestamp > cutoff
        }

    def generate_intervention(self,
                            trigger: InterventionTrigger,
                            user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate intervention based on trigger type."""
        
        if trigger.type == "struggle":
            return self._generate_struggle_intervention(trigger, user_profile)
        elif trigger.type == "boredom":
            return self._generate_boredom_intervention(trigger, user_profile)
        elif trigger.type == "fatigue":
            return self._generate_fatigue_intervention(trigger, user_profile)
        elif trigger.type == "confusion":
            return self._generate_confusion_intervention(trigger, user_profile)
        
        return {
            "type": "general_support",
            "message": "Would you like some assistance with your learning?",
            "suggestions": ["Take a short break", "Review previous material"]
        }

    def _detect_struggle(self, pattern: LearningPattern) -> bool:
        """Detect if user is struggling with the material."""
        if not pattern.accuracy_scores:
            return False
            
        recent_scores = pattern.accuracy_scores[-3:]
        return (len(recent_scores) >= 3 and 
                sum(recent_scores) / len(recent_scores) < self.struggle_threshold)

    def _detect_boredom(self, pattern: LearningPattern) -> bool:
        """Detect if user might be bored."""
        if not pattern.engagement_levels:
            return False
            
        recent_engagement = pattern.engagement_levels[-3:]
        return (len(recent_engagement) >= 3 and
                sum(recent_engagement) / len(recent_engagement) < self.engagement_threshold)

    def _detect_fatigue(self, pattern: LearningPattern) -> bool:
        """Detect signs of learning fatigue."""
        if len(pattern.response_times) < 3:
            return False
            
        recent_times = pattern.response_times[-3:]
        return all(t > self.response_time_threshold for t in recent_times)

    def _detect_confusion(self, pattern: LearningPattern) -> bool:
        """Detect signs of confusion in learning pattern."""
        if len(pattern.hint_usage) < 3:
            return False
            
        recent_hints = pattern.hint_usage[-3:]
        return (sum(recent_hints) / len(recent_hints) > self.confusion_threshold)

    def _calculate_struggle_confidence(self, pattern: LearningPattern) -> float:
        """Calculate confidence in struggle detection."""
        if not pattern.accuracy_scores:
            return 0.0
            
        recent_scores = pattern.accuracy_scores[-3:]
        avg_score = sum(recent_scores) / len(recent_scores)
        trend = recent_scores[-1] - recent_scores[0]
        
        return 1.0 - (avg_score + max(0, trend))

    def _calculate_boredom_confidence(self, pattern: LearningPattern) -> float:
        """Calculate confidence in boredom detection."""
        if not pattern.engagement_levels:
            return 0.0
            
        recent_engagement = pattern.engagement_levels[-3:]
        avg_engagement = sum(recent_engagement) / len(recent_engagement)
        
        return 1.0 - avg_engagement

    def _calculate_fatigue_confidence(self, pattern: LearningPattern) -> float:
        """Calculate confidence in fatigue detection."""
        if len(pattern.response_times) < 3:
            return 0.0
            
        recent_times = pattern.response_times[-3:]
        normalized_times = [min(1.0, t / self.response_time_threshold) 
                          for t in recent_times]
        return sum(normalized_times) / len(normalized_times)

    def _calculate_confusion_confidence(self, pattern: LearningPattern) -> float:
        """Calculate confidence in confusion detection."""
        if len(pattern.hint_usage) < 3:
            return 0.0
            
        recent_hints = pattern.hint_usage[-3:]
        return sum(recent_hints) / len(recent_hints)

    def _base_context(self, pattern: LearningPattern) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        if pattern.objective_id:
            context["objective_id"] = pattern.objective_id
        if pattern.bloom_level:
            context["bloom_level"] = pattern.bloom_level
        if pattern.progress_snapshot:
            context["progress_snapshot"] = pattern.progress_snapshot
        return context

    def _get_struggle_context(self, pattern: LearningPattern, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get context information for struggle intervention."""
        context = {
            "recent_accuracy": pattern.accuracy_scores[-3:],
            "hint_usage": pattern.hint_usage[-3:],
            "avg_response_time": (
                sum(pattern.response_times[-3:]) / 3
                if pattern.response_times else 0
            )
        }
        context.update(base_context)
        return context

    def _get_boredom_context(self, pattern: LearningPattern, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get context information for boredom intervention."""
        context = {
            "engagement_trend": pattern.engagement_levels[-3:],
            "accuracy_level": (
                sum(pattern.accuracy_scores[-3:]) / 3
                if pattern.accuracy_scores else 0
            )
        }
        context.update(base_context)
        return context

    def _get_fatigue_context(self, pattern: LearningPattern, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get context information for fatigue intervention."""
        context = {
            "response_times": pattern.response_times[-3:],
            "session_duration": (
                pattern.timestamps[-1] - pattern.timestamps[0]
                if len(pattern.timestamps) > 1 else timedelta(0)
            ).total_seconds() / 60  # minutes
        }
        context.update(base_context)
        return context

    def _get_confusion_context(self, pattern: LearningPattern, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get context information for confusion intervention."""
        context = {
            "hint_usage_pattern": pattern.hint_usage[-3:],
            "accuracy_trend": pattern.accuracy_scores[-3:]
        }
        context.update(base_context)
        return context

    def _objective_focus(self, context: Dict[str, Any]) -> Optional[str]:
        if not context:
            return None

        progress = context.get("progress_snapshot")
        label: Optional[str] = None
        if isinstance(progress, dict):
            label = (
                progress.get("recommended_objective_label")
                or progress.get("target_skill_label")
                or progress.get("objective_label")
            )
            if not label:
                preview = progress.get("plan_preview")
                if isinstance(preview, dict):
                    next_nodes = preview.get("next_nodes")
                    if isinstance(next_nodes, list) and next_nodes:
                        node = next_nodes[0]
                        if isinstance(node, dict):
                            label = node.get("label")

        objective = context.get("objective_id")
        bloom = context.get("bloom_level")

        pieces: list[str] = []
        if label:
            pieces.append(str(label))
        elif objective:
            pieces.append(str(objective))
        if bloom:
            pieces.append(f"Bloom {bloom}")

        if pieces:
            return " at ".join(pieces) if len(pieces) > 1 else pieces[0]
        return None

    def _generate_struggle_intervention(self,
                                      trigger: InterventionTrigger,
                                      user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intervention for struggling users."""
        focus = self._objective_focus(trigger.context)
        message = "I notice you might be finding this challenging."
        if focus:
            message = f"I notice you might be finding this challenging while working on {focus}."
        return {
            "type": "struggle_support",
            "message": message,
            "suggestions": [
                "Let's break this down into smaller steps",
                "Would you like to review the prerequisite concepts?",
                "I can provide some example problems to practice with"
            ],
            "resources": self._suggest_resources(trigger, user_profile)
        }

    def _generate_boredom_intervention(self,
                                     trigger: InterventionTrigger,
                                     user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intervention for bored users."""
        focus = self._objective_focus(trigger.context)
        message = "Ready for a bigger challenge?"
        if focus:
            message = f"Ready for a bigger challenge on {focus}?"
        return {
            "type": "engagement_boost",
            "message": message,
            "suggestions": [
                "Try this more advanced problem",
                "Here's an interesting real-world application",
                "Would you like to explore a related concept?"
            ],
            "challenge": self._generate_challenge(trigger, user_profile)
        }

    def _generate_fatigue_intervention(self,
                                     trigger: InterventionTrigger,
                                     user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intervention for fatigued users."""
        focus = self._objective_focus(trigger.context)
        message = "Taking regular breaks helps with learning."
        if focus:
            message = f"Taking regular breaks helps with learning—let's pause before continuing with {focus}."
        return {
            "type": "break_suggestion",
            "message": message,
            "suggestions": [
                "Let's take a 5-minute break",
                "Would you like to switch to a different topic?",
                "We can resume this section later"
            ],
            "break_duration": self._suggest_break_duration(trigger)
        }

    def _generate_confusion_intervention(self,
                                       trigger: InterventionTrigger,
                                       user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intervention for confused users."""
        focus = self._objective_focus(trigger.context)
        message = "Let me help clarify this concept."
        if focus:
            message = f"Let me help clarify {focus}."
        return {
            "type": "clarification",
            "message": message,
            "suggestions": [
                "Would you like a different explanation?",
                "I can show you a step-by-step example",
                "Let's identify what's unclear"
            ],
            "explanations": self._generate_alternative_explanations(trigger)
        }

    def _suggest_resources(self,
                          trigger: InterventionTrigger,
                          user_profile: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest relevant learning resources."""
        return [
            {
                "type": "video",
                "title": "Concept Overview",
                "description": "A visual explanation of the key concepts"
            },
            {
                "type": "practice",
                "title": "Practice Problems",
                "description": "Similar problems with step-by-step solutions"
            }
        ]

    def _generate_challenge(self,
                           trigger: InterventionTrigger,
                           user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an appropriate challenge."""
        return {
            "type": "advanced_problem",
            "description": "A more challenging problem to test your skills",
            "difficulty": "advanced",
            "estimated_time": "10-15 minutes"
        }

    def _suggest_break_duration(self, trigger: InterventionTrigger) -> int:
        """Suggest appropriate break duration in minutes."""
        fatigue_level = trigger.confidence
        if fatigue_level > 0.8:
            return 15
        elif fatigue_level > 0.5:
            return 10
        return 5

    def _generate_alternative_explanations(self,
                                         trigger: InterventionTrigger) -> List[Dict[str, str]]:
        """Generate alternative explanations for confused users."""
        return [
            {
                "type": "visual",
                "description": "A diagram-based explanation"
            },
            {
                "type": "example",
                "description": "A real-world example"
            },
            {
                "type": "step_by_step",
                "description": "Detailed step-by-step breakdown"
            }
        ]