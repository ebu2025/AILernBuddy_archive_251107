"""Learning journey orchestrator integrating LLM responses with Bloom's Taxonomy."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

try:  # httpx is optional in some test environments; provide informative failure later.
    import httpx  # type: ignore
except ImportError as exc:  # pragma: no cover - exercised via failure paths in tests
    httpx = None  # type: ignore
    _HTTPX_IMPORT_ERROR = exc
else:
    _HTTPX_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


class DeepSeekGenerationError(RuntimeError):
    """Raised when DeepSeek task generation fails."""


class BloomLevel(Enum):
    """Revised Bloom's Taxonomy levels."""
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"

@dataclass
class LearningObjective:
    """Represents a specific learning objective."""
    id: str
    description: str
    bloom_level: BloomLevel
    prerequisites: List[str]
    subject_area: str
    estimated_time: int  # minutes

@dataclass
class LearnerProgress:
    """Tracks learner's progress through objectives."""
    user_id: str
    completed_objectives: Dict[str, float]  # objective_id -> mastery_level
    current_level: Dict[str, BloomLevel]  # subject_area -> current_bloom_level
    learning_preferences: Dict[str, Any]
    last_assessment: datetime
    intervention_history: List[Dict[str, Any]]

class LearningJourneyOrchestrator:
    """Orchestrates personalized learning journeys using LLM and Bloom's Taxonomy."""
    
    def __init__(self, deepseek_model_config: Dict[str, Any]):
        """Initialize with DeepSeek model configuration."""
        self.model_config = deepseek_model_config
        self.bloom_prompts = self._load_bloom_prompts()
        self.objective_templates = self._load_objective_templates()
        
    def _load_bloom_prompts(self) -> Dict[BloomLevel, str]:
        """Load prompt templates for each Bloom level."""
        return {
            BloomLevel.REMEMBER: """Assess the learner's ability to recall {concept}. 
            Generate questions that focus on memory and recognition.""",
            
            BloomLevel.UNDERSTAND: """Evaluate understanding of {concept}. 
            Create questions that require explanation and interpretation.""",
            
            BloomLevel.APPLY: """Test application of {concept} in new situations. 
            Present scenarios that require using knowledge to solve problems.""",
            
            BloomLevel.ANALYZE: """Challenge analysis skills regarding {concept}. 
            Design tasks that involve breaking down information and finding relationships.""",
            
            BloomLevel.EVALUATE: """Assess ability to evaluate {concept}. 
            Create tasks requiring judgement and critical assessment.""",
            
            BloomLevel.CREATE: """Promote creative thinking about {concept}. 
            Design open-ended tasks that require synthesis and original work."""
        }
    
    def _load_objective_templates(self) -> Dict[str, LearningObjective]:
        """Load predefined learning objective templates."""
        return {
            "bpmn.basics": LearningObjective(
                id="bpmn.basics",
                description="Understand basic BPMN notation and elements",
                bloom_level=BloomLevel.UNDERSTAND,
                prerequisites=[],
                subject_area="bpmn",
                estimated_time=30
            ),
            "test.basic": LearningObjective(
                id="test.basic",
                description="Test basic concept understanding",
                bloom_level=BloomLevel.UNDERSTAND,
                prerequisites=[],
                subject_area="test",
                estimated_time=30,
            ),
            # Add more templates as needed
        }
    
    async def generate_learning_task(self,
                                   objective: LearningObjective,
                                   learner_progress: LearnerProgress) -> Dict[str, Any]:
        """Generate a personalized learning task based on objective and progress."""
        # Get prompt template for current Bloom level
        prompt_template = self.bloom_prompts[objective.bloom_level]
        
        # Personalize prompt based on learner's progress and preferences
        personalized_prompt = self._personalize_prompt(
            prompt_template,
            objective,
            learner_progress
        )
        
        # Build personalization payload for DeepSeek
        learner_profile = self._build_personalization_payload(
            objective,
            learner_progress,
        )

        # Generate task using DeepSeek model
        task = await self._generate_with_llm(
            prompt=personalized_prompt,
            bloom_level=objective.bloom_level.value,
            personalization=learner_profile,
        )
        
        return {
            "task": task,
            "objective_id": objective.id,
            "bloom_level": objective.bloom_level.value,
            "estimated_time": objective.estimated_time,
            "prerequisites": objective.prerequisites
        }
    
    def _personalize_prompt(self,
                           template: str,
                           objective: LearningObjective,
                           progress: LearnerProgress) -> str:
        """Personalize prompt template based on learner's progress and preferences."""
        # Add learning style adaptations
        style_adaptations = self._adapt_to_learning_style(
            progress.learning_preferences.get("learning_style", "visual")
        )
        
        # Add difficulty adjustments based on mastery
        mastery_level = progress.completed_objectives.get(objective.id, 0.0)
        difficulty_adjustments = self._adjust_difficulty(mastery_level)
        
        # Include recent intervention context if relevant
        intervention_context = self._get_relevant_interventions(
            progress.intervention_history,
            objective.subject_area
        )
        
        return f"{template}\n{style_adaptations}\n{difficulty_adjustments}\n{intervention_context}"
    
    def _adapt_to_learning_style(self, learning_style: str) -> str:
        """Add learning style specific instructions to prompt."""
        style_adaptations = {
            "visual": "Include diagrams, charts, or visual representations when possible.",
            "auditory": "Emphasize verbal explanations and discussion-based activities.",
            "kinesthetic": "Incorporate hands-on activities and practical exercises.",
            "reading/writing": "Focus on written explanations and text-based activities."
        }
        return style_adaptations.get(learning_style, "")
    
    def _adjust_difficulty(self, mastery_level: float) -> str:
        """Adjust task difficulty based on mastery level."""
        if mastery_level < 0.3:
            return "Provide additional scaffolding and basic examples."
        elif mastery_level < 0.7:
            return "Include moderate challenges with some guidance."
        else:
            return "Present advanced challenges with minimal scaffolding."
    
    def _get_relevant_interventions(self,
                                  history: List[Dict[str, Any]],
                                  subject_area: str) -> str:
        """Get relevant intervention context from history."""
        recent_interventions = [
            h for h in history
            if h.get("subject_area") == subject_area
            and (datetime.utcnow() - datetime.fromisoformat(h["timestamp"])).days < 7
        ]
        
        if not recent_interventions:
            return ""
            
        return f"Consider recent learning support needs: {recent_interventions[-1]['type']}"
    
    def _build_personalization_payload(
        self,
        objective: LearningObjective,
        progress: LearnerProgress,
    ) -> Dict[str, Any]:
        """Compose learner personalization details for DeepSeek."""

        current_levels: Dict[str, str] = {
            subject: level.value if isinstance(level, BloomLevel) else str(level)
            for subject, level in progress.current_level.items()
        }

        return {
            "learner_id": progress.user_id,
            "preferences": dict(progress.learning_preferences),
            "completed_objectives": dict(progress.completed_objectives),
            "current_bloom_levels": current_levels,
            "last_assessment": progress.last_assessment.isoformat(),
            "recent_interventions": progress.intervention_history[-5:],
            "objective": {
                "id": objective.id,
                "subject_area": objective.subject_area,
                "target_bloom": objective.bloom_level.value,
                "estimated_time": objective.estimated_time,
                "prerequisites": list(objective.prerequisites),
            },
        }

    async def _generate_with_llm(
        self,
        *,
        prompt: str,
        bloom_level: str,
        personalization: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate response using the configured DeepSeek model."""

        if _HTTPX_IMPORT_ERROR is not None or httpx is None:  # pragma: no cover - triggered when httpx missing
            raise DeepSeekGenerationError(
                "httpx is required for DeepSeek integration"
            ) from _HTTPX_IMPORT_ERROR

        api_url = self.model_config.get("api_url")
        if not api_url:
            raise DeepSeekGenerationError("DeepSeek API URL not configured.")

        model_id = self.model_config.get("model_id", "deepseek")
        timeout_seconds = float(self.model_config.get("timeout", 30.0))
        max_retries = int(self.model_config.get("max_retries", 2))
        backoff_seconds = float(self.model_config.get("retry_backoff", 0.5))

        headers: Dict[str, str] = {}
        if api_key := self.model_config.get("api_key"):
            headers["Authorization"] = f"Bearer {api_key}"
        extra_headers = self.model_config.get("headers")
        if isinstance(extra_headers, dict):
            headers.update({str(k): str(v) for k, v in extra_headers.items()})

        payload = {
            "model": model_id,
            "prompt": prompt,
            "bloom_level": bloom_level,
            "personalization": personalization,
        }

        httpx_module = httpx  # local alias for typing clarity
        http_status_error = getattr(httpx_module, "HTTPStatusError", Exception)
        request_error = getattr(httpx_module, "RequestError", Exception)
        timeout_error = getattr(httpx_module, "TimeoutException", request_error)

        last_exception: Optional[Exception] = None
        attempt_count = max(0, max_retries) + 1

        for attempt_index in range(attempt_count):
            attempt_number = attempt_index + 1
            start_time = perf_counter()
            client = httpx_module.AsyncClient(timeout=timeout_seconds)
            try:
                response = await client.post(
                    api_url,
                    json=payload,
                    headers=headers or None,
                )
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                data = response.json()
                latency_ms = int((perf_counter() - start_time) * 1000)
                normalized = self._normalize_deepseek_response(
                    data,
                    fallback_bloom=bloom_level,
                    latency_ms=latency_ms,
                    model_id=model_id,
                )
                logger.info(
                    "DeepSeek task generated in %d ms using model %s (attempt %d/%d)",
                    latency_ms,
                    model_id,
                    attempt_number,
                    attempt_count,
                )
                return normalized
            except http_status_error as exc:
                latency_ms = int((perf_counter() - start_time) * 1000)
                status = getattr(getattr(exc, "response", None), "status_code", "unknown")
                logger.warning(
                    "DeepSeek HTTP error %s for model %s (attempt %d/%d, %d ms): %s",
                    status,
                    model_id,
                    attempt_number,
                    attempt_count,
                    latency_ms,
                    exc,
                )
                last_exception = exc
            except (timeout_error, request_error, ValueError, TypeError, KeyError) as exc:
                latency_ms = int((perf_counter() - start_time) * 1000)
                logger.warning(
                    "DeepSeek request failed for model %s (attempt %d/%d, %d ms): %s",
                    model_id,
                    attempt_number,
                    attempt_count,
                    latency_ms,
                    exc,
                )
                last_exception = exc
            finally:
                try:
                    await client.aclose()
                except Exception:  # pragma: no cover - closing failures are logged but ignored
                    logger.debug("DeepSeek client close failed", exc_info=True)

            if attempt_index < attempt_count - 1:
                await asyncio.sleep(backoff_seconds * (2 ** attempt_index))

        error_detail = (
            f"DeepSeek generation failed after {attempt_count} attempts for model {model_id}."
        )
        raise DeepSeekGenerationError(error_detail) from last_exception

    def _normalize_deepseek_response(
        self,
        payload: Dict[str, Any],
        *,
        fallback_bloom: str,
        latency_ms: int,
        model_id: str,
    ) -> Dict[str, Any]:
        """Normalize DeepSeek response payload for downstream consumption."""

        if not isinstance(payload, dict):
            raise DeepSeekGenerationError("DeepSeek response must be a JSON object.")

        data_section = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        task_section = payload.get("task") if isinstance(payload.get("task"), dict) else {}
        if not task_section:
            task_section = data_section.get("task") if isinstance(data_section, dict) else {}
        if not isinstance(task_section, dict):
            task_section = {}

        content_candidates = [
            task_section.get("content"),
            task_section.get("text"),
            task_section.get("generated"),
            payload.get("content"),
            data_section.get("content") if isinstance(data_section, dict) else None,
        ]
        content = next((c for c in content_candidates if isinstance(c, str) and c.strip()), None)
        if content is None:
            raise DeepSeekGenerationError("DeepSeek response missing task content.")

        bloom_candidates = [
            task_section.get("bloom_level"),
            task_section.get("bloom"),
            payload.get("bloom_level"),
            data_section.get("bloom_level") if isinstance(data_section, dict) else None,
            payload.get("meta", {}).get("bloom_level") if isinstance(payload.get("meta"), dict) else None,
        ]
        bloom_value = next(
            (b for b in bloom_candidates if isinstance(b, str) and b.strip()),
            fallback_bloom,
        )

        metadata: Dict[str, Any] = {
            "model": model_id,
            "latency_ms": latency_ms,
            "source": "deepseek",
        }
        for meta_candidate in (
            payload.get("metadata"),
            payload.get("meta"),
            task_section.get("metadata"),
            data_section.get("metadata") if isinstance(data_section, dict) else None,
        ):
            if isinstance(meta_candidate, dict):
                metadata.update(meta_candidate)

        metadata["bloom_level"] = bloom_value

        scoring_candidates = [
            task_section.get("scoring"),
            metadata.get("scoring"),
            payload.get("scoring"),
            data_section.get("scoring") if isinstance(data_section, dict) else None,
        ]
        scoring = next((s for s in scoring_candidates if isinstance(s, dict)), None)
        if scoring is None:
            alt_scoring = next(
                (s for s in scoring_candidates if isinstance(s, (list, str))),
                None,
            )
            if isinstance(alt_scoring, list):
                scoring = {"hints": alt_scoring}
            elif isinstance(alt_scoring, str):
                scoring = {"hint": alt_scoring}
            else:
                scoring = {}

        return {
            "content": content,
            "bloom_level": bloom_value,
            "scoring": scoring,
            "metadata": metadata,
        }
    
    def update_learner_progress(self,
                              progress: LearnerProgress,
                              objective_id: str,
                              assessment_result: float) -> LearnerProgress:
        """Update learner progress based on assessment results."""
        progress.completed_objectives[objective_id] = assessment_result
        progress.last_assessment = datetime.utcnow()
        
        # Update Bloom level if mastery is sufficient
        if assessment_result >= 0.8:
            templates = getattr(self, "objective_templates", None)
            if not templates:
                templates = self._load_objective_templates()

            current_objective = templates.get(objective_id) if isinstance(templates, dict) else None
            if current_objective is not None:
                subject_area = current_objective.subject_area
            else:
                subject_area = next(iter(progress.current_level.keys()), None)

            if subject_area:
                current_level = progress.current_level.get(
                    subject_area,
                    BloomLevel.REMEMBER
                )

                # Progress to next level if available
                levels = list(BloomLevel)
                current_idx = levels.index(current_level)
                target_idx = (
                    levels.index(current_objective.bloom_level)
                    if current_objective is not None
                    else current_idx
                )
                next_idx = min(len(levels) - 1, max(current_idx, target_idx) + 1)
                if next_idx > current_idx:
                    progress.current_level[subject_area] = levels[next_idx]

        return progress
    
    def get_next_objectives(self,
                          progress: LearnerProgress,
                          subject_area: str,
                          limit: int = 3) -> List[LearningObjective]:
        """Get recommended next objectives based on learner's progress."""
        available_objectives = [
            obj for obj in self._load_objective_templates().values()
            if obj.subject_area == subject_area
            and all(
                progress.completed_objectives.get(prereq, 0.0) >= 0.8
                for prereq in obj.prerequisites
            )
        ]
        
        # Sort by prerequisite completion and current Bloom level
        current_level = progress.current_level.get(subject_area, BloomLevel.REMEMBER)
        sorted_objectives = sorted(
            available_objectives,
            key=lambda obj: (
                obj.bloom_level == current_level,
                -len([p for p in obj.prerequisites if p in progress.completed_objectives])
            )
        )
        
        return sorted_objectives[:limit]