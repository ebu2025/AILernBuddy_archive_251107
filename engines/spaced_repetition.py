"""Spaced repetition system for optimized learning retention."""

from __future__ import annotations

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import math

@dataclass
class ReviewItem:
    concept_id: str
    last_review: datetime
    next_review: datetime
    mastery_level: float
    consecutive_correct: int
    topic: str
    bloom_level: str

class SpacedRepetitionScheduler:
    def __init__(self):
        # SM-2 algorithm parameters
        self.intervals = [1, 3, 7, 14, 30, 60, 120]  # Days between reviews
        self.ease_factors = {
            0: 1.3,  # Difficult
            1: 1.5,  # Good
            2: 1.8,  # Easy
            3: 2.0   # Very Easy
        }
        self.minimum_interval = 1  # Minimum interval in days
        self.maximum_interval = 120  # Maximum interval in days

    def calculate_next_review(self,
                            concept_id: str,
                            mastery_level: float,
                            last_review: datetime,
                            performance_rating: int,
                            consecutive_correct: int) -> datetime:
        """Calculate next review date using modified SM-2 algorithm."""
        
        # Get base interval based on mastery level
        base_interval = self._get_base_interval(mastery_level)
        
        # Apply ease factor based on performance
        ease_factor = self.ease_factors.get(performance_rating, 1.5)
        
        # Calculate interval with consecutive success bonus
        interval_days = base_interval * ease_factor * (1 + (consecutive_correct * 0.1))
        
        # Clamp interval to bounds
        interval_days = max(self.minimum_interval, 
                          min(self.maximum_interval, interval_days))
        
        return last_review + timedelta(days=round(interval_days))

    def _get_base_interval(self, mastery_level: float) -> float:
        """Get base interval days based on mastery level."""
        index = int(mastery_level * (len(self.intervals) - 1))
        return self.intervals[index]

    def get_due_reviews(self,
                       reviews: List[ReviewItem],
                       current_time: Optional[datetime] = None) -> List[ReviewItem]:
        """Get list of items due for review."""
        if current_time is None:
            current_time = datetime.utcnow()
            
        return [item for item in reviews 
                if item.next_review <= current_time]

    def update_review_schedule(self,
                             item: ReviewItem,
                             performance_rating: int,
                             review_time: Optional[datetime] = None) -> ReviewItem:
        """Update review schedule based on performance."""
        if review_time is None:
            review_time = datetime.utcnow()

        # Update consecutive correct count
        if performance_rating >= 2:  # Good or better performance
            consecutive_correct = item.consecutive_correct + 1
        else:
            consecutive_correct = 0

        # Calculate next review time
        next_review = self.calculate_next_review(
            concept_id=item.concept_id,
            mastery_level=item.mastery_level,
            last_review=review_time,
            performance_rating=performance_rating,
            consecutive_correct=consecutive_correct
        )

        # Create updated review item
        return ReviewItem(
            concept_id=item.concept_id,
            last_review=review_time,
            next_review=next_review,
            mastery_level=item.mastery_level,
            consecutive_correct=consecutive_correct,
            topic=item.topic,
            bloom_level=item.bloom_level
        )

    def generate_review_insights(self, 
                               reviews: List[ReviewItem]) -> Dict[str, Any]:
        """Generate insights about review patterns and mastery progress."""
        if not reviews:
            return {"status": "No review data available"}

        total_items = len(reviews)
        mastery_levels = [r.mastery_level for r in reviews]
        avg_mastery = sum(mastery_levels) / total_items

        # Calculate review distribution
        now = datetime.utcnow()
        due_today = len([r for r in reviews if r.next_review.date() == now.date()])
        due_this_week = len([r for r in reviews 
                            if r.next_review.date() <= (now + timedelta(days=7)).date()])

        return {
            "total_items": total_items,
            "average_mastery": round(avg_mastery, 2),
            "due_today": due_today,
            "due_this_week": due_this_week,
            "mastery_distribution": self._calculate_mastery_distribution(reviews),
            "review_load": self._calculate_review_load(reviews)
        }

    def _calculate_mastery_distribution(self, 
                                      reviews: List[ReviewItem]) -> Dict[str, int]:
        """Calculate distribution of mastery levels."""
        distribution = {
            "beginner": 0,      # 0.0-0.3
            "intermediate": 0,   # 0.3-0.7
            "advanced": 0,       # 0.7-0.9
            "mastered": 0       # 0.9-1.0
        }

        for review in reviews:
            if review.mastery_level < 0.3:
                distribution["beginner"] += 1
            elif review.mastery_level < 0.7:
                distribution["intermediate"] += 1
            elif review.mastery_level < 0.9:
                distribution["advanced"] += 1
            else:
                distribution["mastered"] += 1

        return distribution

    def _calculate_review_load(self, 
                             reviews: List[ReviewItem]) -> Dict[str, int]:
        """Calculate review load for next 7 days."""
        now = datetime.utcnow()
        review_load = {}
        
        for i in range(7):
            date = (now + timedelta(days=i)).date()
            review_load[date.isoformat()] = len(
                [r for r in reviews if r.next_review.date() == date]
            )
            
        return review_load

    def suggest_daily_review_plan(self,
                                reviews: List[ReviewItem],
                                available_time_minutes: int = 30) -> Dict[str, Any]:
        """Generate a daily review plan based on available time."""
        due_items = self.get_due_reviews(reviews)
        
        # Estimate 5 minutes per review
        items_possible = min(len(due_items), available_time_minutes // 5)
        
        # Prioritize items
        prioritized_items = sorted(
            due_items,
            key=lambda x: (x.mastery_level, x.next_review)  # Lower mastery first
        )[:items_possible]

        return {
            "total_due": len(due_items),
            "recommended_reviews": items_possible,
            "estimated_time": items_possible * 5,
            "items": [
                {
                    "concept_id": item.concept_id,
                    "topic": item.topic,
                    "bloom_level": item.bloom_level,
                    "mastery_level": item.mastery_level
                }
                for item in prioritized_items
            ]
        }