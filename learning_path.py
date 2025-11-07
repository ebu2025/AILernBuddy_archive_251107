"""Adaptive learning path manager for Bloom-level progression."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional

import db
from bloom_levels import BLOOM_LEVELS, BloomLevelConfigError


_LOGGER = logging.getLogger(__name__)
_PROMPT_VARIANT_FALLBACK = os.getenv("PROMPT_VARIANT", "socratic")
_GLOBAL_PREFERENCES_SUBJECT = "__global__"


def _log_json(event: str, payload: Dict[str, Any]) -> None:
    """Emit structured JSON logs for downstream thesis instrumentation."""

    record = {"event": event, **payload}
    try:
        message = json.dumps(record, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        fallback = {
            "event": event,
            "error": "serialization_failed",
            "payload_repr": repr(payload),
        }
        message = json.dumps(fallback, ensure_ascii=False, sort_keys=True)
    _LOGGER.info(message)


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


def _dedupe_str_list(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _normalise_preferences(preferences: Mapping[str, Any]) -> Dict[str, Any]:
    normalised: Dict[str, Any] = {}
    additional = preferences.get("additional") if isinstance(preferences, Mapping) else None
    for key, value in preferences.items():
        if key == "additional":
            continue
        if value in (None, ""):
            continue
        if isinstance(value, Mapping):
            nested = {
                str(nk): nv
                for nk, nv in value.items()
                if nv not in (None, "", [], {})
            }
            if nested:
                normalised[str(key)] = nested
        elif isinstance(value, (list, tuple, set)):
            deduped = _dedupe_str_list(value)
            if deduped:
                normalised[str(key)] = deduped
        else:
            normalised[str(key)] = value

    if isinstance(additional, Mapping):
        for add_key, add_value in additional.items():
            if add_value in (None, "", [], {}):
                continue
            normalised.setdefault(str(add_key), add_value)

    modalities_source: list[str] = []
    for modality_key in ("modalities", "preferred_modalities"):
        value = normalised.get(modality_key)
        if isinstance(value, list):
            modalities_source.extend(value)
    if modalities_source:
        deduped_modalities = _dedupe_str_list(modalities_source)
        normalised["modalities"] = deduped_modalities
        normalised["preferred_modalities"] = [mod.lower() for mod in deduped_modalities]

    return normalised


def _merge_preferences(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any],
) -> Dict[str, Any]:
    base: Dict[str, Any] = {}
    if isinstance(existing, Mapping):
        for key, value in existing.items():
            base[str(key)] = value

    normalised_incoming = _normalise_preferences(incoming)
    for key, value in normalised_incoming.items():
        if isinstance(value, list):
            previous = base.get(key)
            combined = []
            if isinstance(previous, list):
                combined.extend(previous)
            combined.extend(value)
            base[key] = _dedupe_str_list(combined)
        elif isinstance(value, Mapping):
            previous_mapping = base.get(key)
            merged_mapping: Dict[str, Any] = {}
            if isinstance(previous_mapping, Mapping):
                merged_mapping.update(previous_mapping)
            for nested_key, nested_value in value.items():
                merged_mapping[nested_key] = nested_value
            base[key] = merged_mapping
        else:
            base[key] = value

    return _normalise_preferences(base)


try:
    _DEFAULT_SEQUENCE = BLOOM_LEVELS.sequence() or ("K1", "K2", "K3", "K4", "K5", "K6")
except BloomLevelConfigError:
    _DEFAULT_SEQUENCE = ("K1", "K2", "K3", "K4", "K5", "K6")

try:
    _LOWEST_LEVEL = BLOOM_LEVELS.lowest_level()
except BloomLevelConfigError:
    _LOWEST_LEVEL = _DEFAULT_SEQUENCE[0]

if TYPE_CHECKING:
    from schemas import AssessmentResult


@dataclass
class LearningPathRecommendation:
    """Snapshot returned after an adaptive update."""

    current_level: str
    recommended_level: str
    action: str
    confidence: float
    reason: str
    reason_code: str
    evidence: Dict[str, Any]
    progress_by_level: Dict[str, float]


class AdaptiveLearningPathManager:
    """Rule-based adaptive controller for Bloom-level learning paths.

    The manager maintains a floating mastery score (0-1) per Bloom level and
    updates it using assessment signals (correctness, confidence) as well as a
    rough proxy for learning speed. When a level passes the promotion
    threshold the next Bloom level is unlocked; when it drops below the
    regression threshold the learner is encouraged to revisit a lower level.
    """

    def __init__(
        self,
        *,
        promotion_threshold: float = 0.82,
        regression_threshold: float = 0.35,
        base_step: float = 0.12,
        fast_response_threshold: float = 35.0,
        slow_response_threshold: float = 180.0,
    ) -> None:
        if not 0 < regression_threshold < promotion_threshold < 1:
            raise ValueError("Thresholds must satisfy 0 < regression < promotion < 1")
        self.promotion_threshold = promotion_threshold
        self.regression_threshold = regression_threshold
        self.base_step = base_step
        self.fast_response_threshold = fast_response_threshold
        self.slow_response_threshold = slow_response_threshold
        self._sequence = _DEFAULT_SEQUENCE

    # ------------------------------------------------------------------
    def update_from_assessment(
        self,
        assessment: "AssessmentResult",
        *,
        response_time_seconds: Optional[float] = None,
    ) -> Optional[LearningPathRecommendation]:
        """Convenience helper for ``AssessmentResult`` objects."""

        bloom_level = assessment.bloom_level or _LOWEST_LEVEL
        score = float(assessment.score)
        correct = score >= 0.7
        confidence = float(assessment.confidence or 0.0)
        latency = response_time_seconds
        if latency is None and assessment.latency_ms is not None:
            latency = max(0.0, float(assessment.latency_ms) / 1000.0)
        return self.update_learning_path(
            user_id=assessment.user_id,
            subject_id=assessment.domain,
            bloom_level=bloom_level,
            correct=correct,
            confidence=confidence,
            response_time_seconds=latency,
            evidence={
                "assessment_id": assessment.item_id,
                "score": score,
                "raw_confidence": assessment.confidence,
                "source": str(assessment.source),
            },
        )

    # ------------------------------------------------------------------
    def update_learning_path(
        self,
        *,
        user_id: str,
        subject_id: str,
        bloom_level: str,
        correct: bool,
        confidence: float,
        response_time_seconds: Optional[float] = None,
        evidence: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Optional[LearningPathRecommendation]:
        """Update the adaptive state for ``(user_id, subject_id)``."""

        if not user_id or not subject_id:
            return None

        timer_start = perf_counter()
        prompt_variant = os.getenv("PROMPT_VARIANT", _PROMPT_VARIANT_FALLBACK)

        elo_before: Optional[float] = None
        elo_after: Optional[float] = None
        theta_getter = getattr(db, "get_theta", None)
        if callable(theta_getter):
            try:
                elo_before = float(theta_getter(user_id, subject_id))
            except Exception:
                elo_before = None

        normalized_confidence = max(0.0, min(1.0, confidence))
        state = self._load_state(user_id, subject_id)
        state.pop("updated_at", None)
        if preferences:
            # Incoming (neue) Präferenzen sollen gewinnen
            state["preferences"] = _merge_preferences(state.get("preferences"), preferences)
        level_scores = state.setdefault("levels", {})
        level = bloom_level if bloom_level in self._sequence else _LOWEST_LEVEL
        current_value = float(level_scores.get(level, 0.0))
        previous_level = state.get("current_level", level)
        if previous_level not in self._sequence:
            previous_level = level
        previous_idx = self._sequence.index(previous_level)

        adjustment = self.base_step * (1.0 + normalized_confidence)
        adjustment *= 1 if correct else -1

        speed_factor = self._speed_factor(response_time_seconds)
        adjustment *= speed_factor

        updated_value = max(0.0, min(1.0, current_value + adjustment))
        level_scores[level] = round(updated_value, 4)

        idx = self._sequence.index(level)
        action = "stabilise"
        recommended_level = previous_level
        reason_code = "stabilise_monitor"

        event_evidence = self._prepare_event_evidence(
            level=level,
            normalized_confidence=normalized_confidence,
            correct=correct,
            response_time_seconds=response_time_seconds,
            source_evidence=evidence,
        )

        reason = "Fortschritt stabilisiert – weitere Beobachtung empfohlen."

        if updated_value >= self.promotion_threshold and idx < len(self._sequence) - 1:
            if normalized_confidence < 0.6:
                action = "hold"
                recommended_level = previous_level
                reason_code = "hold_low_confidence"
                reason = "Beförderung zurückgestellt – Vertrauen ist noch zu niedrig."
            else:
                recommended_level = self._sequence[idx + 1]
                action = "promote"
                reason_code = "promote_high_mastery"
                reason = (
                    "Hohe Genauigkeit und Sicherheit auf dem aktuellen Bloom-Level. "
                    "Nächstes Niveau wird freigeschaltet."
                )
        elif updated_value <= self.regression_threshold and idx > 0:
            recommended_level = self._sequence[idx - 1]
            action = "review"
            reason_code = "review_low_mastery"
            reason = (
                "Antworten deuten auf Schwierigkeiten hin – Festigung auf dem vorherigen Bloom-Level empfohlen."
            )
        elif correct:
            reason_code = "stabilise_correct"
            reason = "Antwort korrekt – Fortschritt wird stabil beobachtet."
        else:
            reason_code = "stabilise_incorrect"
            reason = "Antwort unsicher – weitere Übungen auf aktuellem Niveau empfohlen."

        evidence_note = self._summarise_evidence(event_evidence)
        if evidence_note:
            reason = f"{reason} {evidence_note}"

        target_idx = self._sequence.index(recommended_level)
        if target_idx - previous_idx > 1:
            recommended_level = self._sequence[previous_idx + 1]
        elif previous_idx - target_idx > 1:
            recommended_level = self._sequence[previous_idx - 1]

        state["current_level"] = recommended_level
        state.setdefault("history", []).append(
            {
                "bloom_level": level,
                "delta": round(updated_value - current_value, 4),
                "correct": correct,
                "confidence": normalized_confidence,
                "response_time_seconds": response_time_seconds,
            }
        )
        state["history"] = state["history"][-50:]
        state["last_reason"] = reason
        state["last_reason_code"] = reason_code
        state["last_evidence"] = event_evidence

        self._record_adjustment_metadata(
            state,
            previous_level=previous_level,
            recommended_level=recommended_level,
            action=action,
            session_id=session_id,
        )

        db.upsert_learning_path_state(user_id, subject_id, state)
        db.log_learning_path_event(
            user_id,
            subject_id,
            bloom_level=level,
            action=action,
            reason=reason,
            reason_code=reason_code,
            confidence=normalized_confidence,
            evidence=event_evidence,
        )
        db.upsert_bloom_progress(
            user_id,
            subject_id,
            current_level=recommended_level,
            reason=reason,
            average_score=updated_value,
            attempts_considered=len(state.get("history", [])),
            k_level=None,
        )

        if callable(theta_getter):
            try:
                elo_after = float(theta_getter(user_id, subject_id))
            except Exception:
                elo_after = elo_before

        processing_ms = round((perf_counter() - timer_start) * 1000.0, 3)
        latency_seconds = (
            None if response_time_seconds is None else float(response_time_seconds)
        )

        log_payload: Dict[str, Any] = {
            "user_id": user_id,
            "subject_id": subject_id,
            "prompt_variant": prompt_variant,
            "bloom_level_before": previous_level,
            "bloom_level_after": recommended_level,
            "bloom_value_before": round(current_value, 4),
            "bloom_value_after": round(updated_value, 4),
            "elo_before": elo_before,
            "elo_after": elo_after,
            "confidence": round(normalized_confidence, 3),
            "decision": action,
            "reason": reason,
            "reason_code": reason_code,
            "latency_seconds": latency_seconds,
            "processing_ms": processing_ms,
            "evidence": event_evidence,
            "progress_snapshot": {
                lvl: round(val, 4) for lvl, val in level_scores.items()
            },
        }
        _log_json("learning_path_decision", log_payload)

        return LearningPathRecommendation(
            current_level=level,
            recommended_level=recommended_level,
            action=action,
            confidence=round(normalized_confidence, 3),
            reason=reason,
            reason_code=reason_code,
            evidence=event_evidence,
            progress_by_level={lvl: round(val, 4) for lvl, val in level_scores.items()},
        )

    # ------------------------------------------------------------------
    def get_state(self, user_id: str, subject_id: str) -> Dict[str, Any]:
        return self._load_state(user_id, subject_id)

    # ------------------------------------------------------------------
    def persist_preferences(
        self,
        user_id: str,
        preferences: Mapping[str, Any],
        *,
        subject_ids: Optional[Iterable[str]] = None,
    ) -> None:
        """Persistiert globale Präferenzen und lässt Subjekt-Overrides unangetastet.

        Wir speichern die globalen Präferenzen unter __global__.
        Subjekt-States werden NICHT mit globalen Werten befüllt, damit sie nicht veralten.
        Die effektive Sicht entsteht zur Laufzeit in _load_state (global als Basis, Subjekt-Overrides drüber).
        """
        if not user_id:
            return
        if not isinstance(preferences, Mapping):
            return

        normalized = _normalise_preferences(preferences)
        if not normalized:
            return

        # 1) Global aktualisieren
        global_state = self._load_state(user_id, _GLOBAL_PREFERENCES_SUBJECT)
        global_state.pop("updated_at", None)
        global_state["preferences"] = normalized
        db.upsert_learning_path_state(user_id, _GLOBAL_PREFERENCES_SUBJECT, global_state)

        # 2) Optional: bestehende Subjekt-Präferenzen nur normalisieren (keine Fusion mit global!)
        if subject_ids:
            for subject_id in {str(subject) for subject in subject_ids if subject}:
                state = self._load_state(user_id, subject_id)
                state.pop("updated_at", None)
                overrides = state.get("preferences") if isinstance(state.get("preferences"), Mapping) else {}
                state["preferences"] = _normalise_preferences(overrides)
                db.upsert_learning_path_state(user_id, subject_id, state)

    # ------------------------------------------------------------------
    def _load_state(self, user_id: str, subject_id: str) -> Dict[str, Any]:
        raw_state = db.get_learning_path_state(user_id, subject_id) or {}
        updated_at = raw_state.get("updated_at") if isinstance(raw_state, dict) else None
        state: Dict[str, Any] = {}
        if isinstance(raw_state, dict):
            state.update({k: v for k, v in raw_state.items() if k != "updated_at"})

        state.setdefault("levels", {})
        for level in self._sequence:
            state["levels"].setdefault(level, 0.0)
        state.setdefault(
            "current_level",
            state["levels"].get(_LOWEST_LEVEL, _LOWEST_LEVEL),
        )
        state.setdefault("history", [])
        preferences = state.get("preferences") if isinstance(state.get("preferences"), Mapping) else {}
        state["preferences"] = _normalise_preferences(preferences)

        if subject_id != _GLOBAL_PREFERENCES_SUBJECT:
            try:
                global_state = db.get_learning_path_state(user_id, _GLOBAL_PREFERENCES_SUBJECT) or {}
            except Exception:
                global_state = {}
            global_prefs = (
                global_state.get("preferences")
                if isinstance(global_state, dict)
                else {}
            )
            if isinstance(global_prefs, Mapping):
                # Global als Basis, Subjekt-Overrides drüber (Overrides gewinnen)
                state["preferences"] = _merge_preferences(global_prefs, state.get("preferences"))

        if updated_at is not None:
            state["updated_at"] = updated_at
        return state

    def _speed_factor(self, response_time_seconds: Optional[float]) -> float:
        if response_time_seconds is None:
            return 1.0
        if response_time_seconds <= self.fast_response_threshold:
            return 1.1
        if response_time_seconds >= self.slow_response_threshold:
            return 0.85
        return 1.0

    # ------------------------------------------------------------------
    def _prepare_event_evidence(
        self,
        *,
        level: str,
        normalized_confidence: float,
        correct: bool,
        response_time_seconds: Optional[float],
        source_evidence: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "bloom_level": level,
            "normalized_confidence": round(normalized_confidence, 3),
            "correct": bool(correct),
        }
        if response_time_seconds is not None:
            payload["response_time_seconds"] = float(response_time_seconds)

        items: list[Dict[str, Any]] = []
        if source_evidence:
            for key, value in source_evidence.items():
                if key == "items":
                    continue
                payload[key] = value
            existing_items = source_evidence.get("items")
            if isinstance(existing_items, list):
                for entry in existing_items:
                    if isinstance(entry, dict):
                        items.append(dict(entry))
            assessment_id = source_evidence.get("assessment_id")
            score = source_evidence.get("score")
            if assessment_id is not None or score is not None:
                item_entry: Dict[str, Any] = {}
                if assessment_id is not None:
                    item_entry["id"] = assessment_id
                if score is not None:
                    try:
                        item_entry["score"] = float(score)
                    except (TypeError, ValueError):
                        item_entry["score"] = score
                if item_entry:
                    items.append(item_entry)

        if items:
            payload["items"] = items

        return payload

    # ------------------------------------------------------------------
    def _summarise_evidence(self, evidence: Dict[str, Any]) -> str:
        items = evidence.get("items") if isinstance(evidence, dict) else None
        if not items:
            return ""
        summaries: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            score = item.get("score")
            if item_id and isinstance(score, (int, float)):
                summaries.append(f"{item_id} (Score {score:.2f})")
            elif item_id:
                summaries.append(str(item_id))
        if not summaries:
            return ""
        return f"Evidenz: {', '.join(summaries)}."

    # ------------------------------------------------------------------
    def _record_adjustment_metadata(
        self,
        state: Dict[str, Any],
        *,
        previous_level: str,
        recommended_level: str,
        action: str,
        session_id: Optional[str],
    ) -> None:
        now = datetime.now(timezone.utc)
        if session_id:
            state["last_session_id"] = session_id
        if previous_level == recommended_level:
            return

        state["last_adjustment"] = {
            "timestamp": now.isoformat(),
            "session_id": session_id,
            "session_marker": _session_marker(now),
            "action": action,
            "from_level": previous_level,
            "to_level": recommended_level,
        }


__all__ = [
    "AdaptiveLearningPathManager",
    "LearningPathRecommendation",
]
