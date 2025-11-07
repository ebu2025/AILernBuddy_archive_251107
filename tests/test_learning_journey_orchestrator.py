"""Test cases for the learning journey orchestrator."""

from __future__ import annotations

from datetime import datetime, timedelta
import types

import pytest

from engines import learning_journey_orchestrator as orchestrator_module
from engines.learning_journey_orchestrator import (
    BloomLevel,
    LearningObjective,
    LearnerProgress,
    LearningJourneyOrchestrator,
)


@pytest.fixture
def deepseek_stub(monkeypatch):
    """Patch the DeepSeek httpx client with a controllable stub."""

    calls = []
    responses: list[tuple[int, dict]] = []

    class _StubHTTPStatusError(Exception):
        def __init__(self, message: str, *, request=None, response=None):
            super().__init__(message)
            self.request = request
            self.response = response

    class _StubTimeoutError(Exception):
        pass

    class _StubRequestError(Exception):
        pass

    class _StubResponse:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise _StubHTTPStatusError(
                    f"HTTP {self.status_code}",
                    response=self,
                )

        def json(self):
            return self._payload

    class _StubAsyncClient:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        async def post(self, url, json=None, headers=None):
            if not responses:
                raise AssertionError("DeepSeek stub has no queued responses")
            calls.append(
                {
                    "url": url,
                    "json": json,
                    "headers": headers,
                    "client_kwargs": self.kwargs,
                }
            )
            status_code, payload = responses.pop(0)
            return _StubResponse(status_code, payload)

        async def aclose(self):
            return None

    stub_module = types.SimpleNamespace(
        AsyncClient=_StubAsyncClient,
        HTTPStatusError=_StubHTTPStatusError,
        TimeoutException=_StubTimeoutError,
        RequestError=_StubRequestError,
    )

    monkeypatch.setattr(orchestrator_module, "httpx", stub_module)
    monkeypatch.setattr(orchestrator_module, "_HTTPX_IMPORT_ERROR", None)

    return {"responses": responses, "calls": calls}


@pytest.fixture
def anyio_backend():
    """Force anyio to use asyncio backend for async tests."""

    return "asyncio"

@pytest.fixture
def sample_objective():
    """Create a sample learning objective for testing."""
    return LearningObjective(
        id="test.basic",
        description="Test basic concept understanding",
        bloom_level=BloomLevel.UNDERSTAND,
        prerequisites=[],
        subject_area="test",
        estimated_time=30
    )

@pytest.fixture
def sample_progress():
    """Create a sample learner progress for testing."""
    return LearnerProgress(
        user_id="test_user",
        completed_objectives={},
        current_level={"test": BloomLevel.REMEMBER},
        learning_preferences={"learning_style": "visual"},
        last_assessment=datetime.utcnow(),
        intervention_history=[]
    )

@pytest.fixture
def orchestrator(deepseek_stub):
    """Create a learning journey orchestrator instance."""
    return LearningJourneyOrchestrator(
        deepseek_model_config={
            "model_id": "test-model",
            "api_url": "https://deepseek.invalid/v1/generate",
            "timeout": 5,
            "max_retries": 0,
        }
    )

@pytest.mark.anyio("asyncio")
async def test_generate_learning_task(orchestrator, sample_objective, sample_progress, deepseek_stub):
    """Test generation of personalized learning tasks via DeepSeek."""

    deepseek_stub["responses"].append(
        (
            200,
            {
                "data": {
                    "task": {
                        "content": "Explain the primary idea using a process map.",
                        "bloom_level": "understand",
                        "scoring": {"rubric": "Focus on conceptual clarity."},
                    },
                    "metadata": {"tokens": {"prompt": 128, "completion": 256}},
                }
            },
        )
    )

    task = await orchestrator.generate_learning_task(sample_objective, sample_progress)

    assert isinstance(task, dict)
    assert "task" in task
    assert task["objective_id"] == sample_objective.id
    assert task["bloom_level"] == sample_objective.bloom_level.value
    assert task["estimated_time"] == sample_objective.estimated_time

    assert deepseek_stub["calls"], "DeepSeek client was not invoked"
    call = deepseek_stub["calls"][0]
    payload = call["json"]
    assert payload["model"] == "test-model"
    assert payload["bloom_level"] == sample_objective.bloom_level.value
    assert "understand" in payload["prompt"].lower()
    personalization = payload["personalization"]
    assert personalization["learner_id"] == sample_progress.user_id
    assert personalization["objective"]["id"] == sample_objective.id
    assert personalization["current_bloom_levels"]["test"] == sample_progress.current_level["test"].value

    generated = task["task"]
    assert generated["bloom_level"] == sample_objective.bloom_level.value
    assert generated["content"].startswith("Explain the primary idea")
    assert generated["scoring"]["rubric"] == "Focus on conceptual clarity."
    assert generated["metadata"]["model"] == "test-model"
    assert generated["metadata"]["source"] == "deepseek"


def test_update_learner_progress(orchestrator, sample_progress):
    """Test updating learner progress based on assessment results."""
    objective_id = "test.basic"
    assessment_result = 0.85

    previous_timestamp = sample_progress.last_assessment

    updated_progress = orchestrator.update_learner_progress(
        sample_progress,
        objective_id,
        assessment_result
    )

    assert updated_progress.completed_objectives[objective_id] == assessment_result
    assert updated_progress.last_assessment > previous_timestamp
    # Should progress to next Bloom level due to high score
    assert updated_progress.current_level["test"] == BloomLevel.APPLY

def test_get_next_objectives(orchestrator, sample_progress):
    """Test recommendation of next learning objectives."""
    # Add some completed objectives
    sample_progress.completed_objectives = {
        "test.prereq1": 0.9,
        "test.prereq2": 0.85
    }
    
    next_objectives = orchestrator.get_next_objectives(
        sample_progress,
        subject_area="test",
        limit=2
    )
    
    assert isinstance(next_objectives, list)
    assert len(next_objectives) <= 2
    for obj in next_objectives:
        assert isinstance(obj, LearningObjective)
        assert obj.subject_area == "test"
        # Verify prerequisites are met
        assert all(
            sample_progress.completed_objectives.get(prereq, 0.0) >= 0.8
            for prereq in obj.prerequisites
        )

def test_learning_style_adaptation(orchestrator, sample_objective, sample_progress):
    """Test adaptation to different learning styles."""
    # Test visual learning style
    sample_progress.learning_preferences["learning_style"] = "visual"
    prompt = orchestrator._personalize_prompt(
        "Base prompt",
        sample_objective,
        sample_progress
    )
    assert "diagrams" in prompt.lower() or "visual" in prompt.lower()
    
    # Test auditory learning style
    sample_progress.learning_preferences["learning_style"] = "auditory"
    prompt = orchestrator._personalize_prompt(
        "Base prompt",
        sample_objective,
        sample_progress
    )
    assert "verbal" in prompt.lower() or "discussion" in prompt.lower()

def test_difficulty_adjustment(orchestrator):
    """Test difficulty adjustments based on mastery level."""
    # Low mastery
    low_mastery_adjustment = orchestrator._adjust_difficulty(0.2)
    assert "scaffolding" in low_mastery_adjustment.lower()
    
    # Medium mastery
    medium_mastery_adjustment = orchestrator._adjust_difficulty(0.5)
    assert "moderate" in medium_mastery_adjustment.lower()
    
    # High mastery
    high_mastery_adjustment = orchestrator._adjust_difficulty(0.8)
    assert "advanced" in high_mastery_adjustment.lower()

def test_intervention_context(orchestrator, sample_progress):
    """Test inclusion of relevant intervention context."""
    recent_time = datetime.utcnow() - timedelta(days=3)
    old_time = datetime.utcnow() - timedelta(days=10)
    
    # Add some intervention history
    sample_progress.intervention_history = [
        {
            "type": "struggle",
            "subject_area": "test",
            "timestamp": recent_time.isoformat()
        },
        {
            "type": "boredom",
            "subject_area": "test",
            "timestamp": old_time.isoformat()
        }
    ]
    
    context = orchestrator._get_relevant_interventions(
        sample_progress.intervention_history,
        "test"
    )
    
    # Should only include recent intervention
    assert "struggle" in context.lower()
    assert "boredom" not in context.lower()