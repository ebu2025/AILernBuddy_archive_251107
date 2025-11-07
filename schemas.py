"""Pydantic schemas for validated model outputs and helper utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

__all__ = [
    "PendingOps",
    "RubricCriterion",
    "AssessmentStepEvaluation",
    "AssessmentErrorPattern",
    "AssessmentResult",
    "LearnerModel",
    "ChatResponse",
    "LearningPatternRequest",
    "InterventionResponse",
    "parse_json_safe",
]

class LearningPatternRequest(BaseModel):
    """Request model for learning pattern analysis."""
    response_times: List[float]
    accuracy_scores: List[float]
    engagement_levels: List[float]
    hint_usage: List[float]
    timestamps: List[datetime]
    objective_id: str | None = None
    bloom_level: str | None = None
    progress_snapshot: Dict[str, Any] | None = None

class InterventionResponse(BaseModel):
    """Response model for learning interventions."""
    type: str
    confidence: float
    detected_at: datetime
    context: Dict[str, Any]
    intervention: Dict[str, Any]


class PendingOps(BaseModel):
    op_id: str
    type: str
    inputs: Dict[str, Any]
    decision: str
    rationale: str
    next_action: Dict[str, Any] | None = None
    timestamps: Dict[str, Any]


class RubricCriterion(BaseModel):
    id: str
    score: float


class AssessmentStepEvaluation(BaseModel):
    step_id: str = Field(description="Identifier of the reasoning or solution step that was evaluated.")
    subskill: str | None = Field(
        default=None,
        description="Optional fine-grained skill that the step targets (e.g., algebra.linear_equations.balance).",
    )
    outcome: Literal["correct", "incorrect", "hint", "skipped"] = Field(
        description="Outcome assigned to the step after evaluation.",
    )
    score_delta: float | None = Field(
        default=None,
        description="Optional delta or fractional score awarded for the step.",
    )
    hint: str | None = Field(
        default=None,
        description="Hint that was provided after the step, if any, to maintain a Socratic dialogue chain.",
    )
    feedback: str | None = Field(
        default=None,
        description="Qualitative feedback or rubric justification for the step outcome.",
    )
    diagnosis: Literal["conceptual", "procedural", "careless", "none"] | None = Field(
        default=None,
        description="Optional micro-diagnosis highlighting the primary issue observed in the step.",
    )


class AssessmentErrorPattern(BaseModel):
    code: str = Field(description="Stable identifier for the recurring error or misconception pattern.")
    description: str | None = Field(
        default=None,
        description="Human-readable description of the observed error pattern.",
    )
    subskill: str | None = Field(
        default=None,
        description="Optional subskill associated with the error pattern.",
    )
    occurrences: int = Field(
        default=1,
        ge=1,
        description="How often the pattern was observed within the assessment dialogue.",
    )


class AssessmentResult(BaseModel):
    user_id: str
    domain: str
    item_id: str
    bloom_level: str
    response: str
    self_assessment: str | None = Field(
        default=None,
        description="Optional learner-authored reflection captured alongside the attempt.",
    )
    score: float
    rubric_criteria: list[RubricCriterion] = Field(default_factory=list)
    model_version: str
    prompt_version: str
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    confidence: float = Field(
        default=0.0,
        description="Tutor-assigned confidence for the score; only >=0.5 updates progression.",
    )
    diagnosis: Literal["conceptual", "procedural", "careless", "none"] | None = Field(
        default=None,
        description=(
            "Tutor-classified root cause for learner errors: conceptual misunderstanding, "
            "procedural slip, careless mistake, or 'none' when no misconception is observed."
        ),
    )
    source: Literal["direct", "self_check", "heuristic", "pending"] = Field(
        default="direct",
        description=(
            "Assessment provenance: direct=model JSON, self_check=post-hoc regrade, "
            "heuristic=micro-check, pending=queued for review. See ADL/xAPI guidance."
        ),
    )
    step_evaluations: list[AssessmentStepEvaluation] = Field(
        default_factory=list,
        description="Ordered micro-evaluations of intermediate reasoning steps, including Socratic hints.",
    )
    error_patterns: list[AssessmentErrorPattern] = Field(
        default_factory=list,
        description="Aggregated misconception patterns distilled from the step evaluations.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LearnerPreferences(BaseModel):
    modalities: list[str] = Field(
        default_factory=list,
        description="Preferred instructional modalities (e.g., video, text, practice).",
    )
    pacing: str | None = Field(
        default=None,
        description="Learner pace preference (e.g., slow, medium, fast).",
    )
    language_level: str | None = Field(
        default=None,
        description="Self-reported proficiency band for the learner's primary study language.",
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Preferred languages for instruction.",
    )
    time_windows: list[str] = Field(
        default_factory=list,
        description="Preferred study windows or availability notes.",
    )
    additional: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form key/value pairs for any extra learner preferences.",
    )


class LearnerBloomBand(BaseModel):
    lower: str = Field(description="Lower bound of the Bloom band, inclusive.")
    upper: str | None = Field(
        default=None,
        description="Upper bound of the Bloom band, inclusive when provided.",
    )


class LearnerGoal(BaseModel):
    goal_id: str | None = Field(
        default=None,
        description="Stable identifier for the goal, if available.",
    )
    description: str = Field(description="Human readable goal statement for the learner.")
    target_skills: list[str] = Field(
        default_factory=list,
        description="List of skill identifiers this goal is targeting.",
    )
    target_bloom_band: LearnerBloomBand | None = Field(
        default=None,
        description="Optional Bloom taxonomy band with 'lower'/'upper' levels.",
    )
    due_date: datetime | None = Field(
        default=None,
        description="Optional deadline associated with the learning goal.",
    )


class LearnerSkillState(BaseModel):
    skill_id: str = Field(description="Identifier of the tracked skill or competency.")
    proficiency: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Normalized proficiency estimate for the skill (0–1).",
    )
    bloom_band: LearnerBloomBand | None = Field(
        default=None,
        description="Bloom taxonomy band representing demonstrated mastery for the skill.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model confidence in the proficiency estimate (0–1).",
    )
    last_updated: datetime | None = Field(
        default=None,
        description="Timestamp of the last proficiency update.",
    )
    confidence_updated_at: datetime | None = Field(
        default=None,
        description="Timestamp of the last confidence update, if tracked separately.",
    )
    target_success_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Target probability of answering correctly used for cold-start diagnostics.",
    )
    confidence_interval_lower: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Lower bound of the estimated success probability interval for diagnostics.",
    )
    confidence_interval_upper: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Upper bound of the estimated success probability interval for diagnostics.",
    )
    confidence_interval_width: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Width of the success probability interval tracked during calibration.",
    )


class LearnerMisconception(BaseModel):
    misconception_id: str | None = Field(
        default=None,
        description="Identifier for tracking the misconception across updates.",
    )
    skill_id: str | None = Field(
        default=None,
        description="Skill associated with the misconception, if scoped.",
    )
    description: str = Field(
        description="Description of the misconception or knowledge gap.",
    )
    severity: str | None = Field(
        default=None,
        description="Qualitative severity label (e.g., low, medium, high).",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="References to evidence supporting the misconception diagnosis.",
    )
    last_seen: datetime | None = Field(
        default=None,
        description="Last time the misconception was observed or confirmed.",
    )


class LearnerModel(BaseModel):
    user_id: str = Field(description="User identifier for the learner model.")
    goals: list[LearnerGoal] = Field(
        default_factory=list,
        description="Current set of learner goals used for personalization.",
    )
    skills: list[LearnerSkillState] = Field(
        default_factory=list,
        description="Per-skill proficiency, Bloom band, and confidence estimates.",
    )
    preferences: LearnerPreferences = Field(
        default_factory=LearnerPreferences,
        description="Stated learner preferences and contextual constraints.",
    )
    misconceptions: list[LearnerMisconception] = Field(
        default_factory=list,
        description="Known misconceptions that should influence instruction.",
    )
    history_summary: str | None = Field(
        default=None,
        description="Concise narrative of recent learner progress or noteworthy events.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when this learner model snapshot was last updated.",
    )


class ChatResponse(BaseModel):
    answer: str | None = None
    bloom_level: str | None = None
    assessment: Dict[str, Any] | None = None
    assessment_result: Dict[str, Any] | None = None
    diagnosis: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    microcheck_question: str | None = None
    microcheck_expected: str | None = None
    microcheck_answer_key: str | None = None
    microcheck_given: str | None = None
    microcheck_score: float | None = Field(default=None, ge=0.0, le=1.0)
    microcheck_rubric: Dict[str, Any] | list[Any] | None = None
    history_update: Any | None = None
    action: Any | None = None
    db_ops: list[Dict[str, Any]] | None = None
    timestamp: str | None = None
    self_assessment: str | None = None

    model_config = {
        "extra": "allow",
    }


_T = TypeVar("_T", bound=BaseModel)


def _find_first_json_object(text: str) -> tuple[str, int, int]:
    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == "{" and (idx == 0 or text[idx - 1] != "\\"):
                depth += 1
            elif char == "}" and (idx == 0 or text[idx - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        json.loads(candidate)
                    except Exception:
                        break
                    return candidate, start, idx + 1
        start = text.find("{", start + 1)
    raise ValueError("No JSON object found in provided text")


def parse_json_safe(text: str, model: Type[_T]) -> _T:
    """Parse ``text`` into ``model`` with a fallback JSON extraction pass."""

    first_error: Exception | None = None
    try:
        return model.model_validate_json(text)
    except (ValidationError, ValueError, TypeError) as exc:
        first_error = exc

    try:
        snippet, _, end = _find_first_json_object(text)
    except ValueError:
        if first_error:
            raise first_error
        raise

    trailing = text[end:]
    if trailing.strip():
        if isinstance(first_error, ValidationError):
            raise first_error
        raise ValueError("Trailing content detected after JSON object")

    try:
        return model.model_validate_json(snippet)
    except Exception:
        if first_error:
            raise first_error
        raise

