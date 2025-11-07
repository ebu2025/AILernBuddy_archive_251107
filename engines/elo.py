import math
import threading
from typing import Any, Dict

import db

class EloEngine:
    def __init__(self, k: float = 0.6):
        self.k = k
        self._lock = threading.Lock()  # Thread safety for concurrent updates

    @staticmethod
    def predict_success(theta: float, difficulty: float) -> float:
        """Return the success probability for a learner/item pairing."""

        return 1.0 / (1.0 + math.exp(-(theta - difficulty)))

    def placement_band(self, theta: float) -> str:
        """Coarsely categorise ability for reporting/analytics."""

        if theta < -0.4:
            return "intro"
        if theta > 0.4:
            return "stretch"
        return "core"

    def apply_penalty(self, user_id: str, skill: str, *, penalty: float = 0.25) -> Dict[str, Any]:
        """Apply a fixed penalty to a learner's theta for calibration."""

        theta_before = db.get_theta(user_id, skill)
        step = abs(float(penalty))
        theta_after = theta_before - step
        db.set_theta(user_id, skill, theta_after)
        return {
            "user_id": user_id,
            "skill": skill,
            "theta_before": theta_before,
            "theta_after": theta_after,
            "penalty": -step,
        }

    def update(self, user_id: str, skill: str, item_id: str, correct: int) -> Dict[str, Any]:
        with self._lock:  # Ensure atomic updates
            theta = db.get_theta(user_id, skill)
            items = db.list_items(skill=skill, limit=1)
            beta = items[0]["difficulty"] if items else 0.0
            p = self.predict_success(theta, beta)
            delta = self.k * (correct - p)
            new_theta = theta + delta
            db.set_theta(user_id, skill, new_theta)
        return {
            "theta_before": theta,
            "theta_after": new_theta,
            "confidence_before": p,
            "confidence_after": self.predict_success(new_theta, beta),
            "placement_band": self.placement_band(new_theta),
        }
