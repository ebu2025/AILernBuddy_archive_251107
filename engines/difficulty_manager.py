"""Dynamic difficulty adjustment for optimal learning challenge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import re

@dataclass
class DifficultyParameters:
    complexity_level: float  # 0.0 to 1.0
    time_pressure: float    # 0.0 to 1.0
    hint_level: float      # 0.0 to 1.0
    detail_required: float # 0.0 to 1.0

@dataclass
class UserPerformanceMetrics:
    accuracy: float        # 0.0 to 1.0
    response_time: float   # seconds
    hint_usage: float     # 0.0 to 1.0
    confidence: float     # 0.0 to 1.0

class DifficultyManager:
    def __init__(self):
        self.complexity_patterns = {
            "simpler": [
                (r"analyze|evaluate|synthesize", "understand"),
                (r"complex|complicated", "straightforward"),
                (r"multiple factors", "main factors"),
            ],
            "harder": [
                (r"understand|describe", "analyze"),
                (r"straightforward|simple", "complex"),
                (r"basic", "advanced"),
            ]
        }
        
        self.bloom_level_adjustments = {
            "K1": {"up": "K2", "down": "K1"},
            "K2": {"up": "K3", "down": "K1"},
            "K3": {"up": "K4", "down": "K2"},
            "K4": {"up": "K5", "down": "K3"},
            "K5": {"up": "K6", "down": "K4"},
            "K6": {"up": "K6", "down": "K5"},
        }

    def adjust_difficulty(self,
                        question: str,
                        user_performance: UserPerformanceMetrics,
                        current_params: DifficultyParameters,
                        current_bloom_level: str) -> Dict[str, Any]:
        """Adjust question difficulty based on user performance."""
        
        # Calculate overall performance score
        performance_score = self._calculate_performance_score(user_performance)
        
        # Determine adjustment direction
        if performance_score > 0.8:
            adjustment = "increase"
        elif performance_score < 0.4:
            adjustment = "decrease"
        else:
            adjustment = "maintain"

        # Generate new parameters
        new_params = self._adjust_parameters(current_params, adjustment)
        
        # Adjust the question
        adjusted_question = self._modify_question_complexity(
            question, 
            adjustment,
            current_bloom_level
        )
        
        # Determine new Bloom level
        new_bloom_level = self._adjust_bloom_level(
            current_bloom_level, 
            performance_score
        )

        return {
            "adjusted_question": adjusted_question,
            "new_parameters": new_params,
            "new_bloom_level": new_bloom_level,
            "adjustment_type": adjustment,
            "performance_score": performance_score
        }

    def _calculate_performance_score(self, 
                                   metrics: UserPerformanceMetrics) -> float:
        """Calculate overall performance score from metrics."""
        weights = {
            "accuracy": 0.4,
            "confidence": 0.3,
            "response_time": 0.2,
            "hint_usage": 0.1
        }
        
        # Normalize response time (assume 300 seconds is maximum expected time)
        normalized_time = max(0, 1 - (metrics.response_time / 300))
        
        # Calculate weighted score
        score = (
            (metrics.accuracy * weights["accuracy"]) +
            (metrics.confidence * weights["confidence"]) +
            (normalized_time * weights["response_time"]) +
            ((1 - metrics.hint_usage) * weights["hint_usage"])
        )
        
        return min(1.0, max(0.0, score))

    def _adjust_parameters(self,
                         current: DifficultyParameters,
                         adjustment: str) -> DifficultyParameters:
        """Adjust difficulty parameters based on performance."""
        
        def adjust_value(value: float, direction: str) -> float:
            step = 0.1
            if direction == "increase":
                return min(1.0, value + step)
            elif direction == "decrease":
                return max(0.0, value - step)
            return value

        return DifficultyParameters(
            complexity_level=adjust_value(current.complexity_level, adjustment),
            time_pressure=adjust_value(current.time_pressure, adjustment),
            hint_level=adjust_value(current.hint_level, 
                                  "decrease" if adjustment == "increase" else "increase"),
            detail_required=adjust_value(current.detail_required, adjustment)
        )

    def _modify_question_complexity(self,
                                  question: str,
                                  adjustment: str,
                                  bloom_level: str) -> str:
        """Modify question text to match desired complexity."""
        if adjustment == "maintain":
            return question

        patterns = self.complexity_patterns["harder" if adjustment == "increase" 
                                         else "simpler"]
        
        modified_question = question
        for pattern, replacement in patterns:
            modified_question = re.sub(pattern, replacement, 
                                     modified_question, 
                                     flags=re.IGNORECASE)

        return modified_question

    def _adjust_bloom_level(self,
                          current_level: str,
                          performance_score: float) -> str:
        """Adjust Bloom's taxonomy level based on performance."""
        if performance_score > 0.8:
            return self.bloom_level_adjustments[current_level]["up"]
        elif performance_score < 0.4:
            return self.bloom_level_adjustments[current_level]["down"]
        return current_level

    def generate_question_variants(self,
                                 base_question: str,
                                 current_level: str) -> Dict[str, str]:
        """Generate easier and harder variants of a question."""
        return {
            "easier": self._modify_question_complexity(
                base_question, "decrease", current_level
            ),
            "current": base_question,
            "harder": self._modify_question_complexity(
                base_question, "increase", current_level
            )
        }

    def suggest_next_challenge(self,
                             user_performance_history: List[UserPerformanceMetrics],
                             current_level: str) -> Dict[str, Any]:
        """Suggest next challenge level based on performance history."""
        
        # Calculate trend from recent performance
        recent_scores = [
            self._calculate_performance_score(metrics)
            for metrics in user_performance_history[-5:]  # Last 5 attempts
        ]
        
        if not recent_scores:
            return {
                "suggestion": "maintain",
                "confidence": 0.5,
                "reason": "Insufficient performance history"
            }

        avg_score = sum(recent_scores) / len(recent_scores)
        score_trend = recent_scores[-1] - recent_scores[0] \
            if len(recent_scores) > 1 else 0

        return {
            "suggestion": "increase" if avg_score > 0.8 and score_trend >= 0
                        else "decrease" if avg_score < 0.4 and score_trend <= 0
                        else "maintain",
            "confidence": avg_score,
            "reason": self._generate_suggestion_reason(avg_score, score_trend)
        }

    def _generate_suggestion_reason(self,
                                  avg_score: float,
                                  score_trend: float) -> str:
        """Generate explanation for difficulty suggestion."""
        if avg_score > 0.8:
            if score_trend > 0:
                return "Consistently high performance with improving trend"
            else:
                return "High performance but showing slight decline"
        elif avg_score < 0.4:
            if score_trend < 0:
                return "Struggling with current level and declining performance"
            else:
                return "Below target performance but showing improvement"
        else:
            return "Performance within optimal challenge range"