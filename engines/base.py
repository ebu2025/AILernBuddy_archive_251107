from typing import Dict, Any

class BaseEngine:
    def update(self, user_id: str, skill: str, item_id: str, correct: int) -> Dict[str, Any]:
        raise NotImplementedError
