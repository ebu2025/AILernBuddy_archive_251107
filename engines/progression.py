"""Learning progression engine for K-level transitions.

This module implements a lightweight rules engine that analyses
recent quiz attempts and updates the user's working level according
to Learning Progression Analytics (LPA) heuristics. The engine is
intentionally deterministic to provide explainable behaviour while
remaining compatible with offline execution.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import db

from bloom_levels import BLOOM_LEVELS, BloomLevelConfigError
from engines.intervention_system import LearningInterventionSystem, LearningPattern
from process_models.process_mining import generate_process_diagnostics, parse_event_log

_LOGGER = logging.getLogger(__name__)

KLevel = str
try:
    K_LEVEL_SEQUENCE: Sequence[KLevel] = BLOOM_LEVELS.k_level_sequence()
except BloomLevelConfigError:
    K_LEVEL_SEQUENCE = ("K1", "K2", "K3")

try:
    LOWEST_K_LEVEL: KLevel = BLOOM_LEVELS.lowest_level()
except BloomLevelConfigError:
    LOWEST_K_LEVEL = "K1"

sequence = BLOOM_LEVELS.sequence()
if sequence:
    BLOOM_SEQUENCE: Sequence[str] = sequence
else:
    BLOOM_SEQUENCE = ("K1", "K2", "K3", "K4", "K5", "K6")


def _bloom_index(level: str) -> int:
    try:
        return BLOOM_LEVELS.index(level)
    except ValueError:
        return 0


@dataclass
class AttemptSummary:
    """Structured representation of a quiz attempt."""

    attempt_id: int
    activity_id: str
    normalized_score: float
    passed: bool
    pass_threshold: float
    confidence: float
    source: str
    created_at: str
    response_time: Optional[float] = None
    engagement_level: Optional[float] = None
    hint_usage: Optional[float] = None


@dataclass
class ProgressionResult:
    """Outcome of a progression evaluation."""

    previous_level: KLevel
    new_level: KLevel
    changed: bool
    reason: str
    average_score: float
    attempts_considered: int
    intervention_trigger: Optional[Dict[str, Any]] = None


class ProgressionEngine:
    """Rule-based learning progression engine.

    Parameters
    ----------
    advance_threshold:
        Minimum average score required (0-1 range) across the evaluation
        window to promote the learner to the next level.
    regress_threshold:
        Average score threshold below which the learner should revisit
        the previous level, provided enough attempts exist.
    min_attempts:
        Minimum number of recent attempts that must be available before
        a level change is considered.
    window_size:
        Number of recent attempts to inspect. Defaults to 5 to provide a
        balance between responsiveness and stability.
    confidence_boost:
        Optional callable that maps the average score to a confidence
        value for storage in the progress table. A simple linear mapping
        is used when omitted.
    """

    def __init__(
        self,
        advance_threshold: float = 0.8,
        regress_threshold: float = 0.45,
        min_attempts: int = 3,
        window_size: int = 5,
        confidence_boost: Optional[Callable[[float], float]] = None,
        intervention_system: Optional[LearningInterventionSystem] = None,
    ) -> None:
        if not 0.0 < advance_threshold <= 1.0:
            raise ValueError("advance_threshold must be in (0, 1]")
        if not 0.0 <= regress_threshold < advance_threshold:
            raise ValueError("regress_threshold must be lower than advance_threshold")
        if min_attempts <= 0:
            raise ValueError("min_attempts must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_size < min_attempts:
            raise ValueError("window_size must be >= min_attempts")

        self.advance_threshold = float(advance_threshold)
        self.regress_threshold = float(regress_threshold)
        self.min_attempts = int(min_attempts)
        self.window_size = int(window_size)
        self._confidence_boost = confidence_boost or (lambda score: round(score, 3))
        self._min_weight_for_change = 2.0
        self.intervention_system: Optional[LearningInterventionSystem]
        if intervention_system is None:
            self.intervention_system = LearningInterventionSystem()
        else:
            self.intervention_system = intervention_system

    # ----- public API --------------------------------------------------
    def process_attempt(
        self,
        user_id: str,
        subject_id: str,
        activity_id: str,
        score: float,
        max_score: float,
        pass_threshold: float = 0.8,
    ) -> ProgressionResult:
        """Record an attempt and update progression if required."""

        attempt = db.record_quiz_attempt(
            user_id=user_id,
            subject_id=subject_id,
            activity_id=activity_id,
            score=score,
            max_score=max_score,
            pass_threshold=pass_threshold,
        )
        return self.evaluate(user_id, subject_id, last_attempt_id=attempt["attempt_id"])

    def evaluate(
        self,
        user_id: str,
        subject_id: str,
        last_attempt_id: Optional[int] = None,
        *,
        advance_threshold: Optional[float] = None,
        regress_threshold: Optional[float] = None,
    ) -> ProgressionResult:
        """Evaluate the most recent attempts and update the stored level."""

        advance_threshold = (
            float(advance_threshold)
            if advance_threshold is not None
            else float(self.advance_threshold)
        )
        regress_threshold = (
            float(regress_threshold)
            if regress_threshold is not None
            else float(self.regress_threshold)
        )

        attempt_rows = db.list_recent_quiz_attempts(
            user_id=user_id, subject_id=subject_id, limit=self.window_size
        )
        attempts = [
            AttemptSummary(
                attempt_id=int(row["id"]),
                activity_id=row["activity_id"],
                normalized_score=float(row["normalized_score"]),
                passed=bool(row["passed"]),
                pass_threshold=float(row["pass_threshold"]),
                confidence=float(row["confidence"]) if "confidence" in row.keys() else 1.0,
                source=str(row["path"]) if "path" in row.keys() else "direct",
                created_at=row["created_at"],
            )
            for row in attempt_rows
        ]
        considered_attempts = attempts[: self.window_size]
        weights = [self._attempt_weight(a) for a in considered_attempts]
        weight_sum = sum(weights)
        average_recent = (
            sum(a.normalized_score * w for a, w in zip(considered_attempts, weights)) / weight_sum
            if weight_sum > 0
            else 0.0
        )

        intervention_trigger = None
        if self.intervention_system is not None:
            pattern = LearningPattern(
                response_times=[
                    a.response_time for a in considered_attempts if a.response_time is not None
                ],
                accuracy_scores=[a.normalized_score for a in considered_attempts],
                engagement_levels=[
                    a.engagement_level
                    for a in considered_attempts
                    if a.engagement_level is not None
                ],
                hint_usage=[
                    a.hint_usage for a in considered_attempts if a.hint_usage is not None
                ],
                timestamps=[datetime.fromisoformat(a.created_at) for a in considered_attempts],
            )

            # Check for intervention triggers
            intervention_trigger = self.intervention_system.monitor_progress(user_id, pattern)

        intervention_payload = asdict(intervention_trigger) if intervention_trigger else None
        
        progress_row = db.get_user_progress(user_id, subject_id)
        previous_level: KLevel = (
            progress_row["current_level"]
            if progress_row
            else LOWEST_K_LEVEL  # type: ignore[assignment]
        )
        if previous_level not in K_LEVEL_SEQUENCE:
            previous_level = LOWEST_K_LEVEL

        if last_attempt_id is None and progress_row and not _has_new_attempts(progress_row["updated_at"], attempt_rows):
            return ProgressionResult(
                previous_level=previous_level,
                new_level=previous_level,
                changed=False,
                reason="Keine neuen Versuche seit der letzten Auswertung.",
                average_score=round(average_recent, 4),
                attempts_considered=len(considered_attempts),
                intervention_trigger=intervention_payload,
            )

        new_level = previous_level
        reason = "Confidence-weighted evidence not yet sufficient to adjust level."
        promotion_ready = (
            len(considered_attempts) >= self.min_attempts
            and weight_sum >= self._min_weight_for_change
            and average_recent >= advance_threshold
        )
        fail_weight = self._weighted_fail_streak(considered_attempts)
        fail_ratio = (fail_weight / weight_sum) if weight_sum > 0 else 0.0
        regression_needed = (
            len(considered_attempts) >= self.min_attempts
            and fail_weight >= self._min_weight_for_change
            and (
                average_recent <= regress_threshold
                or fail_ratio >= 0.6
            )
        )

        if promotion_ready:
            maybe_next = self._shift_level(previous_level, step=1)
            if maybe_next != previous_level:
                new_level = maybe_next
                reason = (
                    "Confidence-weighted average "
                    f"{average_recent:.2f} with total weight {weight_sum:.1f} exceeds the advancement threshold."
                )
        elif regression_needed:
            maybe_prev = self._shift_level(previous_level, step=-1)
            if maybe_prev != previous_level:
                new_level = maybe_prev
                reason = (
                    "Two-weight struggle detected "
                    f"(cumulative fail weight {fail_weight:.1f}, fail ratio {fail_ratio:.2f}); "
                    "revisiting foundations."
                )
        elif len(considered_attempts) < self.min_attempts or weight_sum < self._min_weight_for_change:
            reason = "Awaiting more confident evidence before adjusting level."
        else:
            reason = "Performance stable — maintaining current level under confidence-weighted review."

        confidence = float(self._confidence_boost(average_recent)) if considered_attempts else 0.0
        db.upsert_user_progress(user_id, subject_id, new_level, confidence)
        existing_progress = None
        try:
            existing_progress = db.get_bloom_progress(user_id, subject_id)
        except Exception:
            existing_progress = None

        try:
            session_time = _attempt_timestamp(considered_attempts[0]) if considered_attempts else None
            if (weight_sum <= 0.0 or not considered_attempts) and existing_progress is None:
                bloom_level = None
            else:
                bloom_level = self._score_to_bloom_level(
                    average_recent,
                    new_level,
                    user_id,
                    subject_id,
                    len(considered_attempts),
                    session_time=session_time,
                )
                db.upsert_bloom_progress(
                    user_id,
                    subject_id,
                    bloom_level,
                    reason=reason,
                    average_score=average_recent,
                    attempts_considered=len(considered_attempts),
                    k_level=new_level,
                )
        except Exception:
            pass

        self._update_td_bkt(user_id, subject_id, considered_attempts)

        return ProgressionResult(
            previous_level=previous_level,
            new_level=new_level,
            changed=new_level != previous_level,
            reason=reason,
            average_score=round(average_recent, 4),
            attempts_considered=len(considered_attempts),
            intervention_trigger=intervention_payload,
        )

    # ----- helpers -----------------------------------------------------
    def _attempt_weight(self, attempt: AttemptSummary) -> float:
        """Return the contribution weight for an attempt based on confidence."""

        confidence = attempt.confidence if attempt.confidence is not None else 1.0
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return 0.0

        if not math.isfinite(confidence):
            return 0.0

        confidence = max(0.0, min(1.0, confidence))
        if confidence < 0.5:
            return 0.0
        return confidence

    def _weighted_fail_streak(self, attempts: Sequence[AttemptSummary]) -> float:
        streak = 0.0
        max_streak = 0.0
        for attempt in attempts:
            weight = self._attempt_weight(attempt)
            if attempt.normalized_score < attempt.pass_threshold:
                streak += weight
                max_streak = max(max_streak, streak)
            else:
                streak = 0.0
        return max_streak

    def _update_td_bkt(
        self,
        user_id: str,
        subject_id: str,
        attempts: Sequence[AttemptSummary],
    ) -> None:
        if os.getenv("ENABLE_TD_BKT", "false").lower() != "true":
            return
        if not attempts:
            return

        decay = float(os.getenv("TD_BKT_DECAY", "0.05"))
        learn = float(os.getenv("TD_BKT_LEARN", "0.15"))
        guess = float(os.getenv("TD_BKT_GUESS", "0.2"))
        slip = float(os.getenv("TD_BKT_SLIP", "0.1"))

        try:
            theta = float(db.get_theta(user_id, subject_id))
        except Exception:
            theta = 0.4

        theta = max(0.0, min(1.0, theta))

        for attempt in reversed(attempts):
            weight = self._attempt_weight(attempt)
            if weight <= 0.0:
                continue

            theta *= math.exp(-decay * weight)
            correct = attempt.normalized_score >= attempt.pass_threshold
            prob_correct = theta * (1 - slip) + (1 - theta) * guess
            if correct:
                numerator = prob_correct * theta
                posterior = numerator / prob_correct if prob_correct > 0 else theta
            else:
                numerator = (1 - prob_correct) * theta
                posterior = numerator / (1 - prob_correct) if prob_correct < 1 else theta

            theta = theta + weight * (posterior - theta)
            theta += (1 - theta) * learn * weight
            theta = max(0.0, min(1.0, theta))

        # TD-BKT (time-decayed Bayesian Knowledge Tracing) stays interpretable because
        # its parameters (guess, slip, learn) are explicit and auditable for lecturers.
        try:
            db.set_theta(user_id, subject_id, theta)
        except Exception:
            pass

    def _shift_level(self, current: KLevel, step: int) -> KLevel:
        idx = K_LEVEL_SEQUENCE.index(current)
        new_idx = max(0, min(len(K_LEVEL_SEQUENCE) - 1, idx + step))
        return K_LEVEL_SEQUENCE[new_idx]

    def _score_to_bloom_level(
        self,
        average_score: float,
        base_level: KLevel,
        user_id: str,
        subject_id: str,
        evidence_count: int,
        *,
        session_time: Optional[datetime] = None,
    ) -> str:
        if evidence_count <= 0:
            existing = db.get_bloom_progress(user_id, subject_id)
            if existing:
                return existing["current_level"]
            return BLOOM_SEQUENCE[_bloom_index(base_level)]

        thresholds = BLOOM_LEVELS.thresholds_descending()
        candidate = BLOOM_LEVELS.lowest_level()
        for threshold, label in thresholds:
            if average_score >= threshold:
                candidate = label
                break

        candidate_index = _bloom_index(candidate)
        base_index = _bloom_index(base_level)
        final_index = max(base_index, candidate_index)

        existing = db.get_bloom_progress(user_id, subject_id)
        learning_state = db.get_learning_path_state(user_id, subject_id) or {}
        if existing:
            previous_index = _bloom_index(existing["current_level"])
            if final_index < previous_index:
                final_index = max(final_index, previous_index - 1)
            if candidate_index > previous_index and _within_session_guard(
                learning_state,
                session_time=session_time,
            ):
                final_index = previous_index
        return BLOOM_SEQUENCE[final_index]


class BPMNProgressionStrategy(ProgressionEngine):
    """Progression engine variant that reacts to BPMN process diagnostics.

    The strategy inspects lightweight process-mining metrics (cycle time,
    bottlenecks, dropout rate) to adjust the advancement and regression
    thresholds before delegating to the base :class:`ProgressionEngine`.
    This keeps the promotion logic aligned with operational reality: long
    waiting times or unhandled escalation paths should make promotions a
    little easier while also protecting learners from unnecessary
    regressions triggered by process issues rather than knowledge gaps.
    """

    def __init__(
        self,
        *,
        event_provider: Optional[Callable[[str, str], Iterable[Mapping[str, Any]]]] = None,
        end_activity: Optional[str] = None,
        advance_threshold: float = 0.8,
        regress_threshold: float = 0.45,
        min_attempts: int = 3,
        window_size: int = 5,
        confidence_boost: Optional[Callable[[float], float]] = None,
        bottleneck_high_hours: float = 18.0,
        bottleneck_critical_hours: float = 36.0,
        smoothing: float = 0.4,
    ) -> None:
        super().__init__(
            advance_threshold=advance_threshold,
            regress_threshold=regress_threshold,
            min_attempts=min_attempts,
            window_size=window_size,
            confidence_boost=confidence_boost,
        )
        self._event_provider = event_provider or (lambda _user, _subject: [])
        self._end_activity = end_activity
        self._base_advance_threshold = float(advance_threshold)
        self._base_regress_threshold = float(regress_threshold)
        self._bottleneck_high_hours = max(0.0, float(bottleneck_high_hours))
        self._bottleneck_critical_hours = max(
            self._bottleneck_high_hours, float(bottleneck_critical_hours)
        )
        self._smoothing = max(0.0, min(1.0, float(smoothing)))
        self._last_thresholds: Dict[str, Any] = {
            "advance_threshold": self.advance_threshold,
            "regress_threshold": self.regress_threshold,
            "context": {},
        }
        self._last_diagnostics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    @property
    def last_thresholds(self) -> Dict[str, Any]:
        """Return the most recently applied dynamic thresholds."""

        return dict(self._last_thresholds)

    # ------------------------------------------------------------------
    @property
    def last_diagnostics(self) -> Dict[str, Any]:
        """Expose cached process diagnostics for observability."""

        return dict(self._last_diagnostics)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        user_id: str,
        subject_id: str,
        last_attempt_id: Optional[int] = None,
    ) -> ProgressionResult:
        diagnostics = self._collect_diagnostics(user_id, subject_id)
        dynamic_advance, dynamic_regress, context = self._derive_thresholds(diagnostics)

        result = super().evaluate(
            user_id,
            subject_id,
            last_attempt_id=last_attempt_id,
            advance_threshold=dynamic_advance,
            regress_threshold=dynamic_regress,
        )

        self._last_thresholds = {
            "advance_threshold": dynamic_advance,
            "regress_threshold": dynamic_regress,
            "context": context,
        }

        context_suffix = self._format_contextual_reason(context)
        if context_suffix:
            result = ProgressionResult(
                previous_level=result.previous_level,
                new_level=result.new_level,
                changed=result.changed,
                reason=f"{result.reason} {context_suffix}".strip(),
                average_score=result.average_score,
                attempts_considered=result.attempts_considered,
            )
        return result

    # ------------------------------------------------------------------
    def _collect_diagnostics(self, user_id: str, subject_id: str) -> Dict[str, Any]:
        try:
            raw_events = list(self._event_provider(user_id, subject_id) or [])
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.warning("BPMNProgressionStrategy event provider failed: %s", exc)
            raw_events = []

        if not raw_events:
            self._last_diagnostics = {}
            return {}

        try:
            events = parse_event_log(raw_events)
        except Exception as exc:  # pragma: no cover - resilience over precision
            _LOGGER.warning("Failed to parse process events for BPMN diagnostics: %s", exc)
            self._last_diagnostics = {}
            return {}

        diagnostics = generate_process_diagnostics(
            events,
            end_activity=self._end_activity,
        )
        self._last_diagnostics = diagnostics
        return diagnostics

    # ------------------------------------------------------------------
    def _derive_thresholds(
        self, diagnostics: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, Any]]:
        bottleneck_hours = 0.0
        dropout_rate = 0.0
        mean_cycle = 0.0

        if diagnostics:
            bottlenecks = diagnostics.get("bottlenecks") or []
            if bottlenecks:
                top = bottlenecks[0]
                bottleneck_hours = float(top.get("mean_wait_hours") or 0.0)
            dropout_info = diagnostics.get("dropouts") or {}
            dropout_rate = float(dropout_info.get("rate") or 0.0)
            cycle_info = diagnostics.get("cycle_time_hours") or {}
            mean_cycle = float(cycle_info.get("mean") or 0.0)

        advance_shift = 0.0
        if bottleneck_hours >= self._bottleneck_critical_hours:
            advance_shift -= 0.08
        elif bottleneck_hours >= self._bottleneck_high_hours:
            advance_shift -= 0.05
        elif 0.0 < bottleneck_hours <= max(4.0, self._bottleneck_high_hours / 3):
            advance_shift += 0.02

        if mean_cycle >= 48.0:
            advance_shift -= 0.03
        elif 0.0 < mean_cycle <= 12.0:
            advance_shift += 0.01

        advance_target = self._clamp(
            self._base_advance_threshold + advance_shift,
            0.55,
            0.92,
        )
        advance_threshold = self._blend(self._base_advance_threshold, advance_target)

        regress_adjust = 0.0
        if dropout_rate >= 0.3:
            regress_adjust += 0.07
        elif dropout_rate >= 0.15:
            regress_adjust += 0.04
        if bottleneck_hours >= self._bottleneck_high_hours:
            regress_adjust += 0.02

        regress_target = self._clamp(
            self._base_regress_threshold - regress_adjust,
            0.15,
            advance_threshold - 0.05,
        )
        regress_threshold = self._blend(self._base_regress_threshold, regress_target)

        context = {
            "top_bottleneck_hours": round(bottleneck_hours, 2) if bottleneck_hours else 0.0,
            "dropout_rate": round(dropout_rate, 4),
            "mean_cycle_hours": round(mean_cycle, 2) if mean_cycle else 0.0,
            "advance_threshold": round(advance_threshold, 3),
            "regress_threshold": round(regress_threshold, 3),
            "source": "process_mining",
        }
        return advance_threshold, regress_threshold, context

    # ------------------------------------------------------------------
    def _format_contextual_reason(self, context: Dict[str, Any]) -> str:
        if not context:
            return ""
        fragments = []
        bottleneck = context.get("top_bottleneck_hours")
        if bottleneck:
            fragments.append(f"Bottleneck {bottleneck:.1f}h")
        dropout = context.get("dropout_rate")
        if dropout:
            fragments.append(f"Dropout {dropout * 100:.1f}%")
        cycle = context.get("mean_cycle_hours")
        if cycle:
            fragments.append(f"Zyklus {cycle:.1f}h")
        fragments.append(
            f"Schwellen dyn. {context.get('advance_threshold'):.2f}/{context.get('regress_threshold'):.2f}"
        )
        return "| BPMN-Diagnose: " + ", ".join(fragments)

    # ------------------------------------------------------------------
    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    # ------------------------------------------------------------------
    def _blend(self, base: float, target: float) -> float:
        if self._smoothing <= 0.0:
            return target
        if self._smoothing >= 1.0:
            return target
        return (1.0 - self._smoothing) * base + self._smoothing * target


class MathProgressionStrategy(ProgressionEngine):
    """Math-specific strategy that emphasises Socratic hinting over solutions."""

    def __init__(
        self,
        *,
        matrix_path: Optional[str | Path] = None,
        hint_fail_threshold: int = 2,
        step_window: int = 25,
        step_diagnostics_provider: Optional[
            Callable[[str, str, int], Iterable[Mapping[str, Any]]]
        ] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        resolved_path = (
            Path(matrix_path)
            if matrix_path is not None
            else Path(__file__).resolve().parent.parent / "competencies" / "math.skillmatrix.json"
        )
        self._skill_matrix = self._load_skill_matrix(resolved_path)
        self._hint_fail_threshold = max(1, int(hint_fail_threshold))
        self._step_window = max(1, int(step_window))
        self._step_provider = step_diagnostics_provider or self._default_step_provider
        self._last_hint_plan: Dict[str, Any] = {
            "preferred_intervention": "practice",
            "recent_failures": 0,
            "message": "Noch keine Socratic-Hinweise erforderlich.",
        }

    # ------------------------------------------------------------------
    @property
    def last_hint_plan(self) -> Dict[str, Any]:
        return dict(self._last_hint_plan)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        user_id: str,
        subject_id: str,
        last_attempt_id: Optional[int] = None,
    ) -> ProgressionResult:
        result = super().evaluate(user_id, subject_id, last_attempt_id=last_attempt_id)

        try:
            plan = self._build_hint_plan(user_id, subject_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.warning("MathProgressionStrategy hint planning failed: %s", exc)
            plan = {
                "preferred_intervention": "practice",
                "recent_failures": 0,
                "message": "Hinweiskette konnte nicht berechnet werden.",
            }

        self._last_hint_plan = plan
        if plan.get("preferred_intervention") == "socratic_hint":
            message = plan.get("message", "")
            updated_reason = f"{result.reason} {message}".strip()
            result = ProgressionResult(
                previous_level=result.previous_level,
                new_level=result.new_level,
                changed=result.changed,
                reason=updated_reason,
                average_score=result.average_score,
                attempts_considered=result.attempts_considered,
            )
        return result

    # ------------------------------------------------------------------
    def _build_hint_plan(self, user_id: str, subject_id: str) -> Dict[str, Any]:
        attempt_rows_raw = db.list_recent_quiz_attempts(
            user_id=user_id, subject_id=subject_id, limit=self.window_size
        )
        attempt_rows = [dict(row) for row in attempt_rows_raw]
        failures = [
            row
            for row in attempt_rows
            if not bool(row.get("passed"))
            and float(row.get("confidence") or 0.0) >= 0.5
        ]
        if len(failures) < self._hint_fail_threshold:
            return {
                "preferred_intervention": "practice",
                "recent_failures": len(failures),
                "message": "Fehlerquote unter der Socratic-Hinweis-Schwelle.",
            }

        domain = self._skill_matrix.get("domain") or subject_id
        step_rows: list[Dict[str, Any]] = []
        for entry in self._step_provider(user_id, domain, self._step_window) or []:
            if isinstance(entry, Mapping):
                step_rows.append(dict(entry))
            elif hasattr(entry, "keys"):
                row_dict = {key: entry[key] for key in entry.keys()}
                step_rows.append(row_dict)
        

        failure_counter: Counter[str] = Counter()
        hint_evidence: dict[str, list[Dict[str, Any]]] = defaultdict(list)
        for row in step_rows:
            subskill = (row.get("subskill") or "").strip()
            if not subskill:
                continue
            outcome = str(row.get("outcome") or "").lower()
            diagnosis = str(row.get("diagnosis") or "").lower()
            if outcome in {"incorrect", "hint"} or diagnosis in {"conceptual", "procedural"}:
                failure_counter[subskill] += 1
                hint_evidence[subskill].append(
                    {
                        "step_id": row.get("step_id"),
                        "hint": row.get("hint"),
                        "feedback": row.get("feedback"),
                        "diagnosis": diagnosis or None,
                        "created_at": row.get("created_at"),
                    }
                )

        target_subskill: str = ""
        evidence_count = 0
        if failure_counter:
            target_subskill, evidence_count = failure_counter.most_common(1)[0]
        else:
            inferred = self._infer_skill_from_attempts(failures)
            if inferred:
                target_subskill = inferred
                evidence_count = len(failures)

        parent_skill, skill_data = self._resolve_skill_data(target_subskill)
        methods = list(skill_data.get("recommended_methods", []) or ["Socratic Hints"])
        if "Socratic Hints" not in methods:
            methods.insert(0, "Socratic Hints")

        suspected = None
        if target_subskill and skill_data.get("misconceptions"):
            diag_counter = Counter(
                (entry.get("diagnosis") or "none")
                for entry in hint_evidence.get(target_subskill, [])
                if entry.get("diagnosis")
            )
            dominant = diag_counter.most_common(1)
            if dominant:
                diag_label = dominant[0][0]
                for mis in skill_data.get("misconceptions", []):
                    tags = [str(tag).lower() for tag in mis.get("tags", [])]
                    if diag_label in tags:
                        suspected = mis.get("description")
                        break

        message_parts = [
            f"Socratic Hinweisphase aktiviert: {len(failures)} Fehlversuche in Folge.",
        ]
        if target_subskill:
            message_parts.append(f"Fokus: {target_subskill}")
        if methods:
            method_text = ", ".join(methods)
            message_parts.append(f"Methoden: {method_text}")
        if hint_evidence.get(target_subskill):
            hints_preview = [
                entry.get("hint")
                for entry in hint_evidence[target_subskill]
                if entry.get("hint")
            ]
            if hints_preview:
                message_parts.append(
                    "Letzte Socratic-Hinweise: " + "; ".join(hints_preview[:2])
                )
        if suspected:
            message_parts.append(f"Vermutete Fehlvorstellung: {suspected}")

        return {
            "preferred_intervention": "socratic_hint",
            "recent_failures": len(failures),
            "target_skill": parent_skill,
            "target_subskill": target_subskill or parent_skill,
            "evidence_count": evidence_count,
            "methods": methods,
            "suspected_misconception": suspected,
            "evidence": hint_evidence.get(target_subskill, []),
            "message": " ".join(message_parts).strip(),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _default_step_provider(
        user_id: str, domain: str, limit: int
    ) -> Iterable[Mapping[str, Any]]:
        try:
            return db.list_recent_step_diagnostics(user_id, domain, limit=limit)
        except Exception:
            return []

    # ------------------------------------------------------------------
    def _load_skill_matrix(self, path: Path) -> Dict[str, Any]:
        skills: Dict[str, Any] = {}
        subskills: Dict[str, Dict[str, Any]] = {}
        domain = "math"
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:  # pragma: no cover - configuration guard
            _LOGGER.warning("Math skill matrix not found at %s", path)
            return {"domain": domain, "skills": skills, "subskills": subskills}
        except json.JSONDecodeError:  # pragma: no cover - configuration guard
            _LOGGER.warning("Invalid math skill matrix JSON at %s", path)
            return {"domain": domain, "skills": skills, "subskills": subskills}

        domain = payload.get("domain", domain)
        for skill in payload.get("skills", []):
            skill_id = skill.get("id")
            if not skill_id:
                continue
            skills[skill_id] = skill
            for subskill in skill.get("subskills", []):
                sub_id = subskill.get("id")
                if not sub_id:
                    continue
                subskills[sub_id] = {"skill_id": skill_id, "data": subskill}
        return {"domain": domain, "skills": skills, "subskills": subskills}

    # ------------------------------------------------------------------
    def _resolve_skill_data(self, subskill: str) -> Tuple[str, Dict[str, Any]]:
        mapping = self._skill_matrix.get("subskills", {}).get(subskill)
        if mapping:
            skill_id = mapping.get("skill_id") or subskill
            return skill_id, self._skill_matrix.get("skills", {}).get(skill_id, {})
        return subskill, self._skill_matrix.get("skills", {}).get(subskill, {})

    # ------------------------------------------------------------------
    def _infer_skill_from_attempts(
        self, attempts: Sequence[Mapping[str, Any]]
    ) -> Optional[str]:
        skill_ids = set(self._skill_matrix.get("skills", {}).keys())
        for row in attempts:
            activity = str(row.get("activity_id") or "")
            for skill_id in skill_ids:
                if activity.startswith(skill_id):
                    return skill_id
        return None

def ensure_progress_record(
    user_id: str,
    subject_id: str,
    default_level: Optional[KLevel] = None,
) -> None:
    """Ensure a progress row exists with a default level."""

    if default_level is None:
        default_level = BLOOM_LEVELS.lowest_level()

    row = db.get_user_progress(user_id, subject_id)
    if row is None:
        db.upsert_user_progress(user_id, subject_id, default_level, confidence=0.0)


_ELO_DIAGNOSIS_DELTAS: dict[str, float] = {
    "conceptual": -90.0,
    "procedural": -60.0,
    "careless": -10.0,
    "default": -25.0,
}


def retry_question(
    current_target: float | int | None,
    diagnosis: Optional[str] = None,
    *,
    floor: float = 200.0,
    ceiling: float = 1600.0,
) -> dict[str, float | str | None]:
    """Return a softened ELO target for the next retry based on the diagnosis.

    Parameters
    ----------
    current_target:
        The previous ELO target that was served to the learner.
    diagnosis:
        The misconception category emitted by the tutor (conceptual,
        procedural, careless). ``None`` falls back to the ``"default"``
        adjustment bucket.
    floor / ceiling:
        Clamp the resulting target so it stays within a sensible band.

    Returns
    -------
    dict
        A payload containing the adjusted target, the applied delta and the
        normalised diagnosis label.
    """

    try:
        base_target = float(current_target) if current_target is not None else 0.0
    except (TypeError, ValueError):
        base_target = 0.0

    diag_key = (diagnosis or "").strip().lower()
    delta = _ELO_DIAGNOSIS_DELTAS.get(diag_key, _ELO_DIAGNOSIS_DELTAS["default"])

    unclamped = base_target + delta
    adjusted = max(float(floor), min(float(ceiling), unclamped))
    applied_delta = adjusted - base_target

    _LOGGER.info(
        "retry_question: target_elo adjusted to %.1f (delta %.1f) after %s miss",
        adjusted,
        applied_delta,
        diag_key or "unspecified",
    )

    return {
        "target_elo": adjusted,
        "delta": applied_delta,
        "diagnosis": diag_key or None,
    }


def _has_new_attempts(last_update: Optional[str], attempt_rows) -> bool:
    if not attempt_rows:
        return False
    if not last_update:
        return True
    return any(row["created_at"] > last_update for row in attempt_rows)


def _session_window_minutes() -> float:
    try:
        value = float(os.getenv("LEARNING_SESSION_WINDOW_MINUTES", "45"))
    except (TypeError, ValueError):
        value = 45.0
    return max(1.0, value)


def _session_marker(moment: datetime) -> str:
    window_seconds = _session_window_minutes() * 60.0
    bucket = int(moment.timestamp() // window_seconds)
    return f"bucket:{bucket}"


def _attempt_timestamp(attempt: AttemptSummary) -> Optional[datetime]:
    return _parse_timestamp(attempt.created_at)


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _within_session_guard(
    learning_state: dict[str, Any] | Any,
    *,
    session_time: Optional[datetime],
) -> bool:
    if not isinstance(learning_state, dict):
        return False

    info = learning_state.get("last_adjustment")
    if not isinstance(info, dict):
        return False
    if info.get("action") != "promote":
        return False

    current_marker = _session_marker(session_time) if session_time else None
    last_marker = info.get("session_marker") or info.get("session_id")
    if current_marker and isinstance(last_marker, str) and current_marker == last_marker:
        return True

    if session_time is None:
        return False

    last_timestamp = _parse_timestamp(info.get("timestamp"))
    if not last_timestamp:
        return False

    window_seconds = _session_window_minutes() * 60.0
    delta = abs((session_time - last_timestamp).total_seconds())
    return delta <= window_seconds
