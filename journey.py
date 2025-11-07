"""Learning journey orchestration utilities."""

from __future__ import annotations

import json
import uuid
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence
from collections import Counter, defaultdict

import db
import xapi
from bloom_levels import BLOOM_LEVELS
from engines.elo import EloEngine
from schemas import LearnerBloomBand, LearnerModel, LearnerSkillState
from knowledge_graph import KnowledgeGraph

try:  # pragma: no cover - fallback for misconfigured registries
    _BLOOM_SEQUENCE: Sequence[str] = BLOOM_LEVELS.sequence()
except Exception:  # pragma: no cover
    _BLOOM_SEQUENCE = ("K1", "K2", "K3", "K4", "K5", "K6")

_DEFAULT_BLOOM_LOCK = tuple(_BLOOM_SEQUENCE[:2]) if len(_BLOOM_SEQUENCE) > 1 else tuple(_BLOOM_SEQUENCE)


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

_FALLBACK_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_INACTIVITY_THRESHOLD_HOURS = 48

_SKILL_AREA_ALIASES = {
    "bpmn": "business_process",
    "business": "business_process",
    "business_process": "business_process",
    "process": "business_process",
    "math": "mathematics",
    "mathematics": "mathematics",
    "language": "language",
    "mandarin": "language",
    "german": "language",
}

_DEFAULT_MEDIA_CHANNEL = {
    "business_process": "diagram_tools",
    "mathematics": "plotter",
    "language": "audio",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


def _normalise_skill_area(area: Optional[str]) -> str:
    if not area:
        return "unknown"
    base = str(area)
    candidate = base.split(":", 1)[0]
    if "." in candidate:
        candidate = candidate.split(".", 1)[0]
    candidate = candidate.lower()
    return _SKILL_AREA_ALIASES.get(candidate, candidate)


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    """Best-effort parsing for ISO 8601 timestamps.

    The journey log mixes event timestamps from multiple sources (SQLite defaults,
    API payloads, xAPI statements). This helper normalises them into timezone-aware
    ``datetime`` objects so downstream analytics stay resilient to format drift.
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.strptime(text, ISO_FORMAT)
        return parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    if text.endswith("Z"):
        candidate = text[:-1] + "+00:00"
    else:
        candidate = text
    try:
        parsed = datetime.fromisoformat(candidate)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _event_time_key(event: Dict[str, Any]) -> datetime:
    ts = _parse_iso_timestamp(event.get("recorded_at"))
    return ts or _FALLBACK_EPOCH


def _extract_bloom_level(event: Dict[str, Any]) -> Optional[str]:
    """Find the bloom level attached to an event if one exists."""

    candidates = [event]
    details = event.get("details")
    if isinstance(details, dict):
        candidates.append(details)
        learning_state = details.get("learning_state")
        if isinstance(learning_state, dict):
            candidates.append(learning_state)
    for scope in candidates:
        if not isinstance(scope, dict):
            continue
        for key in ("bloom_level", "target_bloom_level", "target_bloom", "bloom"):
            value = scope.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _event_success(event: Dict[str, Any]) -> Optional[bool]:
    """Infer whether an event resulted in success, failure, or was neutral."""

    details = event.get("details")
    if isinstance(details, dict):
        for key in ("success", "correct", "passed", "completed"):
            if key in details:
                raw = details.get(key)
                if isinstance(raw, bool):
                    return raw
                if isinstance(raw, (int, float)):
                    return bool(raw)
                if isinstance(raw, str):
                    lowered = raw.lower()
                    if lowered in {"true", "yes", "correct", "passed", "complete", "completed"}:
                        return True
                    if lowered in {"false", "no", "incorrect", "failed", "incomplete"}:
                        return False
        outcome = details.get("outcome")
        if isinstance(outcome, str):
            lowered = outcome.lower()
            if lowered in {"correct", "success", "passed"}:
                return True
            if lowered in {"incorrect", "failed", "error"}:
                return False
    score = event.get("score")
    if score is not None:
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            return None
        if score_value >= 0.7:
            return True
        if score_value <= 0.3:
            return False
    return None


def _derive_session_insights(session: Dict[str, Any]) -> Dict[str, Any]:
    events = list(session.get("events") or [])
    sorted_events = sorted(events, key=_event_time_key)
    started_at_dt = _parse_iso_timestamp(session.get("started_at"))
    ended_at_dt = _parse_iso_timestamp(session.get("ended_at"))
    first_event_dt: Optional[datetime] = None
    last_event_dt: Optional[datetime] = None
    bloom_transitions: List[Dict[str, Any]] = []
    encountered_levels: List[str] = []
    last_level: Optional[str] = None
    running_success = 0
    running_failure = 0
    longest_success = 0
    total_failures = 0

    for event in sorted_events:
        event_dt = _parse_iso_timestamp(event.get("recorded_at"))
        if event_dt:
            if first_event_dt is None or event_dt < first_event_dt:
                first_event_dt = event_dt
            if last_event_dt is None or event_dt > last_event_dt:
                last_event_dt = event_dt
        level = _extract_bloom_level(event)
        if level:
            if level not in encountered_levels:
                encountered_levels.append(level)
            if last_level and level != last_level:
                bloom_transitions.append(
                    {
                        "from": last_level,
                        "to": level,
                        "at": event.get("recorded_at"),
                    }
                )
            last_level = level
        outcome = _event_success(event)
        if outcome is True:
            running_success += 1
            running_failure = 0
            if running_success > longest_success:
                longest_success = running_success
        elif outcome is False:
            total_failures += 1
            running_failure += 1
            running_success = 0
        else:
            running_success = 0
            running_failure = 0

    effective_end = ended_at_dt or last_event_dt
    effective_start = started_at_dt or first_event_dt
    duration_seconds: Optional[float] = None
    if effective_start and effective_end and effective_end >= effective_start:
        duration_seconds = (effective_end - effective_start).total_seconds()
    elif effective_start and last_event_dt and last_event_dt >= effective_start:
        duration_seconds = (last_event_dt - effective_start).total_seconds()

    insights = {
        "duration_seconds": round(duration_seconds, 2) if duration_seconds is not None else None,
        "event_count": len(events),
        "first_activity_at": (first_event_dt or started_at_dt).strftime(ISO_FORMAT)
        if (first_event_dt or started_at_dt)
        else None,
        "last_activity_at": (last_event_dt or ended_at_dt or started_at_dt).strftime(ISO_FORMAT)
        if (last_event_dt or ended_at_dt or started_at_dt)
        else None,
        "bloom": {
            "transitions": bloom_transitions,
            "levels_encountered": encountered_levels,
            "last_level": last_level,
        },
        "streaks": {
            "current_success_streak": running_success,
            "current_failure_streak": running_failure,
            "longest_success_streak": longest_success,
            "total_failures": total_failures,
        },
    }
    return insights


def _append_session_events(
    collected: List[Dict[str, Any]], events: Sequence[Dict[str, Any]], session_id: Optional[str]
) -> None:
    for event in events:
        combined = {
            "session_id": session_id,
            "event_type": event.get("event_type"),
            "details": event.get("details", {}),
            "recorded_at": event.get("recorded_at"),
            "score": event.get("score"),
        }
        collected.append(combined)


_BPMN_MISSING_GATEWAY_ALIASES = {
    "missing_gateway",
    "gateway_missing",
    "fehlendes_gateway",
    "kein_gateway",
}
_BPMN_ERROR_HANDLING_ALIASES = {
    "incorrect_error_handling",
    "error_handling_issue",
    "faulty_error_handling",
    "fehlende_escalation",
    "fehlende_eskalation",
    "wrong_escalation",
    "fehlerhafte_fehlerbehandlung",
}
_BPMN_TEXT_SOURCES = ("message", "feedback", "note", "summary", "explanation", "diagnosis")


def _derive_bpmn_feedback(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect BPMN modelling issues such as missing gateways or faulty error handling."""

    missing_gateway_cases: List[Dict[str, Any]] = []
    error_handling_cases: List[Dict[str, Any]] = []
    gateway_mentions = 0
    escalation_mentions = 0
    symbol_mentions = 0

    for event in events:
        details = event.get("details") or {}
        if not isinstance(details, dict):
            details = {}
        text_values: List[str] = []
        issue_code_raw: Optional[str] = None
        for key, value in details.items():
            if isinstance(value, str):
                text_values.append(value.lower())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        text_values.append(item.lower())
            elif isinstance(value, dict):
                for sub_value in value.values():
                    if isinstance(sub_value, str):
                        text_values.append(sub_value.lower())
            if key in {"bpmn_issue", "modeling_issue", "issue_code", "diagnosis_code"}:
                candidate = details.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    issue_code_raw = candidate.strip().lower().replace("-", "_")
        aggregate_text = " ".join(text_values)
        if "gateway" in aggregate_text or "gateway" in (issue_code_raw or ""):
            gateway_mentions += 1
        if "escalation" in aggregate_text or "eskalation" in aggregate_text or "error event" in aggregate_text:
            escalation_mentions += 1
        if "symbol" in aggregate_text or "notation" in aggregate_text:
            symbol_mentions += 1

        normalized_issue = None
        if issue_code_raw:
            if any(alias in issue_code_raw for alias in _BPMN_MISSING_GATEWAY_ALIASES):
                normalized_issue = "missing_gateway"
            elif any(alias in issue_code_raw for alias in _BPMN_ERROR_HANDLING_ALIASES):
                normalized_issue = "incorrect_error_handling"
        if not normalized_issue:
            lowered = aggregate_text.lower()
            if any(term in lowered for term in ["fehlendes gateway", "missing gateway", "gateway fehlt"]):
                normalized_issue = "missing_gateway"
            elif any(term in lowered for term in [
                "fehlerbehandlung", "error handling", "fehler behandlung",
                "escalation", "eskalation", "unhandled error", "unbehandelt"
            ]) and any(term in lowered for term in ["falsch", "fehl", "incorrect", "unhandled"]):
                normalized_issue = "incorrect_error_handling"

        if not normalized_issue:
            gateway_expected = details.get("gateway_expected")
            gateway_present = details.get("gateway_present")
            if isinstance(gateway_expected, bool) and gateway_expected and not gateway_present:
                normalized_issue = "missing_gateway"
            escalation_expected = details.get("escalation_expected")
            escalation_handled = details.get("escalation_handled")
            if normalized_issue is None and isinstance(escalation_expected, bool) and escalation_expected:
                if not isinstance(escalation_handled, bool) or not escalation_handled:
                    normalized_issue = "incorrect_error_handling"

        if normalized_issue:
            sample_note: Optional[str] = None
            for key in _BPMN_TEXT_SOURCES:
                value = details.get(key)
                if isinstance(value, str) and value.strip():
                    sample_note = value.strip()
                    break
            record = {
                "event_type": event.get("event_type"),
                "recorded_at": event.get("recorded_at"),
                "detail": sample_note,
            }
            if normalized_issue == "missing_gateway":
                missing_gateway_cases.append(record)
            elif normalized_issue == "incorrect_error_handling":
                error_handling_cases.append(record)

    feedback: Dict[str, Any] = {
        "issues": [],
        "coverage": {
            "gateway_mentions": gateway_mentions,
            "escalation_mentions": escalation_mentions,
            "symbol_mentions": symbol_mentions,
        },
    }

    if missing_gateway_cases:
        feedback["issues"].append(
            {
                "code": "missing_gateway",
                "occurrences": len(missing_gateway_cases),
                "recommendation": (
                    "Setze ein exklusives oder paralleles Gateway ein, um die Sequenzflüsse wieder zusammenzuführen "
                    "und dokumentiere Trigger sowie Abschlusskriterien am Start- und Endereignis."
                ),
                "samples": missing_gateway_cases[:3],
            }
        )

    if error_handling_cases:
        feedback["issues"].append(
            {
                "code": "incorrect_error_handling",
                "occurrences": len(error_handling_cases),
                "recommendation": (
                    "Überprüfe Eskalations- und Abbruchereignisse und verknüpfe sie mit klaren Serviceaufgaben "
                    "oder Entscheidungsregeln für die Fehlerbehandlung."
                ),
                "samples": error_handling_cases[:3],
            }
        )

    if not feedback["issues"]:
        feedback["issues"].append(
            {
                "code": "no_bpmn_issues_detected",
                "occurrences": 0,
                "recommendation": (
                    "Keine modellierungsspezifischen Auffälligkeiten gefunden – prüfe dennoch regelmäßig Gateways "
                    "und Eskalationspfade gegen die Best-Practice-Checkliste."
                ),
                "samples": [],
            }
        )

    return feedback


def _max_datetime(first: Optional[datetime], second: Optional[datetime]) -> Optional[datetime]:
    if first is None:
        return second
    if second is None:
        return first
    return first if first >= second else second


def _ensure_metadata(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


_OUTCOME_ALIASES = {
    "pass": "success",
    "passed": "success",
    "successful": "success",
    "succeed": "success",
    "mastered": "success",
    "complete": "completed",
    "completed": "completed",
    "finish": "completed",
    "finished": "completed",
    "fail": "failed",
    "failed": "failed",
    "failure": "failed",
    "incorrect": "failed",
    "wrong": "failed",
    "error": "failed",
    "missed": "failed",
    "retry": "in_progress",
    "retrying": "in_progress",
    "progress": "in_progress",
    "in-progress": "in_progress",
    "in_progress": "in_progress",
    "attempt": "in_progress",
    "attempted": "in_progress",
    "attempting": "in_progress",
    "started": "in_progress",
    "starting": "in_progress",
    "pending": "pending",
    "queued": "pending",
    "assigned": "assigned",
    "select": "assigned",
    "selected": "assigned",
}


def _normalize_outcome_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "success" if value else "failed"
    text = str(value).strip().lower()
    if not text:
        return None
    if text in _OUTCOME_ALIASES:
        return _OUTCOME_ALIASES[text]
    return text


def enrich_event_metadata(
    details: Optional[Dict[str, Any]],
    *,
    subject_id: Optional[str] = None,
    skill_id: Optional[str] = None,
    competency_id: Optional[str] = None,
    score: Optional[float] = None,
    event_type: Optional[str] = None,
    outcome: Optional[str] = None,
    status: Optional[str] = None,
    graph: Optional[KnowledgeGraph] = None,
) -> Dict[str, Any]:
    metadata = dict(details or {})

    skill_candidate = (
        skill_id
        or competency_id
        or metadata.get("skill_id")
        or metadata.get("competency_id")
        or metadata.get("skill")
        or metadata.get("topic")
        or subject_id
    )
    if skill_candidate:
        normalized_skill = str(skill_candidate).strip()
        if normalized_skill:
            metadata.setdefault("skill_id", normalized_skill)
            metadata.setdefault("competency_id", normalized_skill)

    node: Optional[Any] = None
    if graph:
        for candidate in (
            metadata.get("competency_id"),
            metadata.get("skill_id"),
            competency_id,
            skill_id,
        ):
            if not candidate:
                continue
            candidate_str = str(candidate)
            node = graph.get_node(candidate_str)
            if node:
                break
            matches = graph.find_nodes(skill_ids=[candidate_str])
            if matches:
                node = matches[0]
                break
            if ":" in candidate_str:
                parts = candidate_str.split(":")
                if len(parts) >= 2:
                    simple_id = parts[1]
                    matches = graph.find_nodes(skill_ids=[simple_id])
                    if matches:
                        node = matches[0]
                        break
        if node:
            metadata.setdefault("competency_id", node.identifier)

    skill_area = metadata.get("skill_area")
    if node:
        skill_area = node.domain
    elif not skill_area:
        skill_area = subject_id
    if skill_area:
        metadata["skill_area"] = str(skill_area)

    media_channel = metadata.get("media_channel")
    if graph:
        cluster: Optional[str] = None
        resource_id = metadata.get("resource_id") or metadata.get("content_id")
        if resource_id:
            lookup = graph.lookup_resource(str(resource_id))
            if lookup:
                _, resource = lookup
                cluster = resource.media_cluster or resource.modality
                metadata.setdefault("resource_title", resource.title)
                metadata.setdefault("resource_uri", resource.uri)
        if not cluster and node:
            clusters = graph.resource_clusters(node.identifier)
            if clusters:
                cluster = max(clusters.items(), key=lambda item: item[1])[0]
        if not cluster:
            domain_key = _normalise_skill_area(str(skill_area)) if skill_area else None
            cluster = _DEFAULT_MEDIA_CHANNEL.get(domain_key or "", media_channel)
        if cluster:
            media_channel = cluster
    if media_channel and str(media_channel).strip():
        metadata["media_channel"] = str(media_channel)

    candidate_outcome: Optional[str] = None
    for candidate in (
        outcome,
        status,
        metadata.get("outcome"),
        metadata.get("status"),
    ):
        if candidate:
            candidate_outcome = candidate
            break

    if candidate_outcome is None:
        success_flag = metadata.get("success")
        if isinstance(success_flag, bool):
            candidate_outcome = "success" if success_flag else "failed"
        elif isinstance(success_flag, str) and success_flag.strip():
            candidate_outcome = success_flag

    if candidate_outcome is None:
        correct_flag = metadata.get("correct")
        if isinstance(correct_flag, bool):
            candidate_outcome = "success" if correct_flag else "failed"

    if candidate_outcome is None:
        completed_flag = metadata.get("completed")
        if isinstance(completed_flag, bool):
            candidate_outcome = "completed" if completed_flag else None

    if candidate_outcome is None and score is not None:
        try:
            normalized_score = float(score)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            normalized_score = None
        if normalized_score is not None and not math.isnan(normalized_score) and not math.isinf(normalized_score):
            if normalized_score > 1.0:
                normalized_score = normalized_score / 100.0
            normalized_score = max(0.0, min(1.0, normalized_score))
            if normalized_score >= 0.7:
                candidate_outcome = "success"
            elif normalized_score <= 0.4:
                candidate_outcome = "failed"
            else:
                candidate_outcome = "in_progress"

    if candidate_outcome is None and event_type:
        lowered = str(event_type).lower()
        if "start" in lowered:
            candidate_outcome = "in_progress"
        elif "select" in lowered:
            candidate_outcome = "assigned"
        elif "calibration" in lowered or "complete" in lowered:
            candidate_outcome = "completed"

    normalized_outcome = _normalize_outcome_label(candidate_outcome)
    if normalized_outcome:
        metadata["outcome"] = normalized_outcome
        metadata.setdefault("status", normalized_outcome)
    else:
        metadata.setdefault("outcome", "unknown")
        metadata.setdefault("status", metadata.get("outcome", "unknown"))

    return metadata


def _band_for_difficulty(value: Optional[float]) -> int:
    try:
        difficulty = float(value) if value is not None else 0.0
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        difficulty = 0.0
    if difficulty <= -0.2:
        return 0  # introductory / scaffolded
    if difficulty >= 0.2:
        return 2  # challenge / stretch
    return 1  # core band


def select_calibration_items(
    skill: Optional[str],
    limit: int = 3,
    *,
    user_id: Optional[str] = None,
    db_module=db,
    item_bank: Optional[Dict[str, Any]] = None,
    elo_engine: Optional[EloEngine] = None,
    target_success: float = 0.65,
) -> List[Dict[str, Any]]:
    """Return a short calibration set targeted at the learner's uncertainty.

    When the learner has low confidence (wide posterior intervals) the selector
    concentrates items near ``target_success`` to rapidly reduce uncertainty.
    As confidence improves it widens the probability bands, intentionally
    sampling easier and harder material for placement decisions. Item metadata
    is sourced from the SQL catalogue first, falling back to JSON/tutor banks.
    """

    effective_limit = max(1, int(limit))
    bank = item_bank if item_bank is not None else getattr(__import__("tutor"), "ITEM_BANK", {})

    topic_key = skill or ""
    engine = elo_engine or EloEngine()
    theta = 0.0
    stored_confidence = 0.0
    stored_ci_width: float | None = None
    stored_target_probability: float | None = None
    if user_id:
        try:
            theta = float(db_module.get_theta(user_id, topic_key or "general"))
        except Exception:
            theta = 0.0
        try:
            progress_row = db_module.get_user_progress(user_id, topic_key or "general")
        except Exception:
            progress_row = None

        def _safe_row_get(row: Any, key: str) -> Any:
            if isinstance(row, dict):
                return row.get(key)
            try:
                return row[key]
            except (KeyError, TypeError, IndexError):  # pragma: no cover - sqlite row fallback
                return getattr(row, key, None)

        if progress_row:
            confidence_raw = _safe_row_get(progress_row, "confidence")
            if confidence_raw is not None:
                try:
                    stored_confidence = float(confidence_raw)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    stored_confidence = 0.0

            ci_width_raw = _safe_row_get(progress_row, "ci_width")
            if ci_width_raw is not None:
                try:
                    stored_ci_width = float(ci_width_raw)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    stored_ci_width = None

            target_prob_raw = _safe_row_get(progress_row, "target_probability")
            if target_prob_raw is not None:
                try:
                    stored_target_probability = float(target_prob_raw)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    stored_target_probability = None

    seen_ids: set[str] = set()
    candidates: List[Dict[str, Any]] = []

    # Pull recent items from the SQL-backed pool first for the requested skill
    for row in db_module.list_items(topic_key or None, limit=effective_limit * 4):
        entry = dict(row) if not isinstance(row, dict) else row
        item_id = str(entry.get("id"))
        if not item_id or item_id in seen_ids:
            continue
        candidates.append(
            {
                "id": item_id,
                "skill": entry.get("skill") or topic_key,
                "difficulty": float(entry.get("difficulty")) if entry.get("difficulty") is not None else None,
                "body": (entry.get("body") or "").strip(),
                "source": "db",
            }
        )
        seen_ids.add(item_id)

    # Supplement with curated item bank entries (JSON-backed) when available
    for row in db_module.list_item_bank(skill_id=topic_key or None, limit=effective_limit * 4):
        entry = dict(row) if not isinstance(row, dict) else row
        item_id = str(entry.get("id"))
        if not item_id or item_id in seen_ids:
            continue
        metadata_json = entry.get("metadata_json")
        metadata: Dict[str, Any] = {}
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except (TypeError, ValueError):
                metadata = {}
        candidates.append(
            {
                "id": item_id,
                "skill": entry.get("skill_id") or topic_key,
                "difficulty": float(entry.get("difficulty")) if entry.get("difficulty") is not None else None,
                "body": (entry.get("stimulus") or "").strip(),
                "bloom_level": entry.get("bloom_level"),
                "metadata": metadata or None,
                "topic": metadata.get("topic"),
                "tags": metadata.get("tags"),
                "diagnosis_focus": metadata.get("diagnosis_focus"),
                "source": "item_bank",
            }
        )
        seen_ids.add(item_id)

    # Finally, dip into the lightweight tutor bank as a guaranteed fallback
    bank_spec = {}
    if topic_key and isinstance(bank, dict):
        bank_spec = bank.get(topic_key, {}) or {}
    if not bank_spec and isinstance(bank, dict):
        bank_spec = next(iter(bank.values()), {}) or {}

    def _collect_tags(*sources: Any) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()
        for source in sources:
            if not source:
                continue
            if isinstance(source, str):
                candidate = source.strip()
                if candidate:
                    lowered = candidate.lower()
                    if lowered not in seen:
                        seen.add(lowered)
                        tags.append(candidate)
                continue
            try:
                iterator = iter(source)
            except TypeError:
                candidate = str(source).strip()
                if candidate:
                    lowered = candidate.lower()
                    if lowered not in seen:
                        seen.add(lowered)
                        tags.append(candidate)
                continue
            for entry in iterator:
                if entry is None:
                    continue
                candidate = str(entry).strip()
                if not candidate:
                    continue
                lowered = candidate.lower()
                if lowered not in seen:
                    seen.add(lowered)
                    tags.append(candidate)
        return tags

    spec_topic = bank_spec.get("topic") if isinstance(bank_spec, dict) else None
    spec_tags = bank_spec.get("tags") if isinstance(bank_spec, dict) else None
    spec_metadata = bank_spec.get("metadata") if isinstance(bank_spec, dict) else None

    for item in bank_spec.get("items", []) if isinstance(bank_spec, dict) else []:
        item_id = str(item.get("id"))
        if not item_id or item_id in seen_ids:
            continue
        metadata: Dict[str, Any] = {}
        if isinstance(spec_metadata, dict):
            metadata.update(spec_metadata)
        if spec_topic and "topic" not in metadata:
            metadata["topic"] = spec_topic
        if isinstance(item.get("metadata"), dict):
            metadata.update(item["metadata"])
        item_topic = item.get("topic")
        if item_topic and "topic" not in metadata:
            metadata["topic"] = item_topic
        tags = _collect_tags(spec_tags, item.get("tags"), metadata.get("tags"))
        if tags:
            metadata["tags"] = tags
        elif "tags" in metadata and not metadata.get("tags"):
            metadata.pop("tags", None)
        diagnosis_focus = item.get("diagnosis_focus")
        if diagnosis_focus:
            metadata["diagnosis_focus"] = diagnosis_focus
        candidates.append(
            {
                "id": item_id,
                "skill": item.get("skill_id") or topic_key,
                "difficulty": float(item.get("difficulty")) if item.get("difficulty") is not None else None,
                "body": (item.get("body") or "").strip(),
                "bloom_level": item.get("bloom_level"),
                "metadata": metadata or None,
                "topic": metadata.get("topic"),
                "tags": metadata.get("tags"),
                "diagnosis_focus": metadata.get("diagnosis_focus"),
                "answer_key": item.get("answer_key"),
                "source": "tutor_bank",
            }
        )
        seen_ids.add(item_id)

    if not candidates:
        return []

    band_targets = {0: -0.6, 1: 0.0, 2: 0.6}
    bands: Dict[int, List[Dict[str, Any]]] = {0: [], 1: [], 2: []}
    for candidate in candidates:
        band = _band_for_difficulty(candidate.get("difficulty"))
        candidate["band"] = band
        diff_value = candidate.get("difficulty")
        if isinstance(diff_value, (int, float)):
            predicted = engine.predict_success(theta, float(diff_value))
            candidate["predicted_success"] = predicted
        else:
            candidate["predicted_success"] = None
        bands[band].append(candidate)

    def _difficulty_for_probability(prob: float) -> float:
        clipped = max(0.01, min(0.99, prob))
        odds = clipped / (1.0 - clipped)
        return theta - math.log(odds)

    def _confidence_tolerance(conf: float, ci_width: Optional[float]) -> float:
        if ci_width is not None:
            if ci_width >= 0.45:
                return 0.1
            if ci_width >= 0.3:
                return 0.16
            return 0.24
        if conf < 0.4:
            return 0.12
        if conf < 0.7:
            return 0.18
        return 0.25

    def _target_probabilities(
        conf: float,
        ci_width: Optional[float],
        base: float,
        cap: int,
    ) -> List[float]:
        base_clipped = max(0.1, min(0.9, base))
        sequence: List[float] = [base_clipped]
        if ci_width is not None:
            if ci_width >= 0.45:
                sequence.extend(
                    [
                        max(0.45, base_clipped - 0.05),
                        min(0.85, base_clipped + 0.05),
                    ]
                )
            elif ci_width >= 0.3:
                sequence.extend(
                    [
                        max(0.3, base_clipped - 0.12),
                        min(0.9, base_clipped + 0.12),
                    ]
                )
            else:
                sequence.extend(
                    [
                        max(0.25, base_clipped - 0.2),
                        min(0.95, base_clipped + 0.2),
                    ]
                )
        else:
            if conf < 0.4:
                sequence.extend([max(0.25, base_clipped - 0.1), min(0.85, base_clipped + 0.1)])
            elif conf < 0.7:
                sequence.extend([min(0.9, base_clipped + 0.15), max(0.3, base_clipped - 0.15)])
            else:
                sequence.extend([min(0.95, base_clipped + 0.2), max(0.25, base_clipped - 0.25)])

        deduped: List[float] = []
        for prob in sequence:
            if prob not in deduped:
                deduped.append(prob)
            if len(deduped) >= cap:
                break
        while len(deduped) < cap:
            deduped.append(base_clipped)
        return deduped[:cap]

    effective_target = stored_target_probability if stored_target_probability is not None else target_success
    tolerance = _confidence_tolerance(stored_confidence, stored_ci_width)
    target_probs = _target_probabilities(
        stored_confidence,
        stored_ci_width,
        effective_target,
        effective_limit,
    )

    selected: List[Dict[str, Any]] = []
    chosen_ids: set[str] = set()

    for target_prob in target_probs:
        target_difficulty = _difficulty_for_probability(target_prob)
        best_entry: Dict[str, Any] | None = None
        best_score: tuple[float, float, float] | None = None
        for entry in candidates:
            item_id = entry.get("id")
            if not item_id or item_id in chosen_ids:
                continue
            predicted = entry.get("predicted_success")
            difficulty = entry.get("difficulty")
            if predicted is None or difficulty is None:
                continue
            diff_prob = abs(predicted - target_prob)
            if diff_prob > tolerance:
                continue
            score = (
                diff_prob,
                abs(float(difficulty) - target_difficulty),
                abs(float(difficulty) - band_targets.get(entry["band"], 0.0)),
            )
            if best_score is None or score < best_score:
                best_entry = entry
                best_score = score
        if best_entry is None:
            continue
        selected.append(best_entry)
        chosen_ids.add(best_entry.get("id"))

    # Relax tolerance progressively if we still need more items
    relaxed_tolerance = tolerance
    while len(selected) < effective_limit and relaxed_tolerance < 0.35:
        relaxed_tolerance += 0.05
        for entry in candidates:
            item_id = entry.get("id")
            if not item_id or item_id in chosen_ids:
                continue
            predicted = entry.get("predicted_success")
            difficulty = entry.get("difficulty")
            if predicted is None or difficulty is None:
                continue
            if abs(predicted - effective_target) > relaxed_tolerance:
                continue
            selected.append(entry)
            chosen_ids.add(item_id)
            if len(selected) >= effective_limit:
                break

    if len(selected) < effective_limit:
        remaining = sorted(
            candidates,
            key=lambda entry: (
                abs((entry.get("predicted_success") or effective_target) - effective_target),
                abs((entry.get("difficulty") or 0.0) - band_targets.get(entry.get("band", 1), 0.0)),
                entry.get("id", ""),
            ),
        )
        for entry in remaining:
            item_id = entry.get("id")
            if not item_id or item_id in chosen_ids:
                continue
            selected.append(entry)
            chosen_ids.add(item_id)
            if len(selected) >= effective_limit:
                break

    for entry in selected:
        entry.setdefault("skill", topic_key)
        entry.setdefault("body", "")
        entry.setdefault("source", "unknown")

    return selected[:effective_limit]


def prepare_diagnostic_calibration(
    user_id: str,
    subject_id: Optional[str],
    *,
    db_module=db,
    elo_engine: Optional[EloEngine] = None,
    min_confidence: float = 0.6,
    penalty_step: float = 0.35,
    max_penalties: int = 6,
    bloom_lock: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Seed learner mastery state for a diagnostic calibration run.

    The routine intentionally decreases ability (theta) in small steps until
    the uncertainty (1 - success probability on a core item) reaches the
    desired confidence threshold. Bloom progression is locked to the first
    two knowledge bands to focus calibration on foundational outcomes.
    """

    if not user_id:
        raise ValueError("user_id required for calibration")

    topic = subject_id or "general"
    engine = elo_engine or EloEngine()
    lock_levels = tuple(bloom_lock) if bloom_lock else _DEFAULT_BLOOM_LOCK
    lock_start = lock_levels[0] if lock_levels else "K1"
    upper_band = lock_levels[-1] if len(lock_levels) > 1 else None
    if upper_band == lock_start:
        upper_band = None

    theta_before = db_module.get_theta(user_id, topic)
    success_prob = engine.predict_success(theta_before, difficulty=0.0)
    confidence_before = max(0.0, min(1.0, 1.0 - success_prob))

    confidence_after = confidence_before
    theta_after = theta_before
    penalties: List[Dict[str, Any]] = []

    # Apply gradual penalties until uncertainty crosses the threshold
    attempts = 0
    target_confidence = max(0.0, min(1.0, float(min_confidence)))
    while confidence_after < target_confidence and attempts < max_penalties:
        attempts += 1
        penalty_result = engine.apply_penalty(user_id, topic, penalty=penalty_step)
        theta_after = penalty_result["theta_after"]
        success_prob = engine.predict_success(theta_after, difficulty=0.0)
        updated_confidence = max(0.0, min(1.0, 1.0 - success_prob))
        penalty_result.update(
            {
                "confidence_before": confidence_after,
                "confidence_after": updated_confidence,
            }
        )
        penalties.append(penalty_result)
        confidence_after = updated_confidence

    confidence_growth = confidence_after - confidence_before

    target_probability = 0.65
    ci_width = max(0.25, min(0.6, confidence_after + 0.15))
    ci_lower = max(0.0, target_probability - ci_width / 2)
    ci_upper = min(1.0, target_probability + ci_width / 2)

    db_module.upsert_user_progress(
        user_id,
        topic,
        lock_start,
        confidence=confidence_after,
        band_lower=lock_start,
        band_upper=upper_band,
        target_probability=target_probability,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_width,
    )
    try:
        db_module.upsert_bloom_progress(user_id, topic, lock_start)
    except Exception:  # pragma: no cover - defensive guard when bloom tables missing
        pass

    placement_band = engine.placement_band(theta_after)

    proficiency_estimate = engine.predict_success(theta_after, difficulty=0.0)

    outcome = {
        "user_id": user_id,
        "subject_id": topic,
        "theta_before": theta_before,
        "theta_after": theta_after,
        "confidence_before": confidence_before,
        "confidence_after": confidence_after,
        "confidence_growth": confidence_growth,
        "penalties_applied": len(penalties),
        "penalty_trace": penalties,
        "placement_band": placement_band,
        "bloom_lock": list(lock_levels),
        "proficiency_estimate": proficiency_estimate,
        "target_success_probability": target_probability,
        "confidence_interval": {
            "center": target_probability,
            "lower": ci_lower,
            "upper": ci_upper,
            "width": ci_width,
        },
    }

    try:
        learner_model = db_module.get_learner_model(user_id)
    except Exception:  # pragma: no cover - fallback when learner model unavailable
        learner_model = LearnerModel(user_id=user_id)

    now = datetime.now(timezone.utc)
    bloom_band = LearnerBloomBand(lower=lock_start, upper=upper_band)
    skill_state = LearnerSkillState(
        skill_id=topic,
        proficiency=proficiency_estimate,
        bloom_band=bloom_band,
        confidence=confidence_after,
        last_updated=now,
        confidence_updated_at=now,
        target_success_probability=target_probability,
        confidence_interval_lower=ci_lower,
        confidence_interval_upper=ci_upper,
        confidence_interval_width=ci_width,
    )

    updated_skills: List[LearnerSkillState] = []
    replaced = False
    for entry in learner_model.skills:
        if entry.skill_id == topic:
            updated_skills.append(skill_state)
            replaced = True
        else:
            updated_skills.append(entry)
    if not replaced:
        updated_skills.append(skill_state)

    updated_model = learner_model.model_copy(update={"skills": updated_skills, "updated_at": now})
    try:
        db_module.update_learner_model(updated_model)
    except Exception:  # pragma: no cover - learner model persistence best-effort
        pass

    try:
        db_module.log_journey_update(user_id, "diagnostic_calibration", outcome)
    except Exception:  # pragma: no cover - diagnostics should not break flow
        pass

    return outcome


def _user_from_session_id(session_id: str) -> Optional[str]:
    if not session_id:
        return None
    if ":" not in session_id:
        return None
    return session_id.split(":", 1)[0]


class LearningJourneyTracker:
    """Coordinate learning sessions, events, and summaries via ``journey_log``."""

    def __init__(
        self,
        db_module=db,
        *,
        history_window: int = 250,
        graph: Optional[KnowledgeGraph] = None,
    ) -> None:
        self._db = db_module
        self._history_window = max(50, history_window)
        self._graph = graph

    # ------------------------------------------------------------------
    def start_session(
        self,
        user_id: str,
        subject_id: Optional[str],
        session_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session_id = f"{user_id}:{uuid.uuid4().hex}"
        started_at = _now_iso()
        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "subject_id": subject_id,
            "session_type": session_type,
            "metadata": _ensure_metadata(metadata),
            "started_at": started_at,
        }
        self._log(user_id, "session_started", payload)
        try:
            xapi.emit(
                user_id,
                "http://adlnet.gov/expapi/verbs/initialized",
                f"session:{session_id}",
                context={
                    "session_type": session_type,
                    "subject": subject_id,
                    "metadata": payload["metadata"],
                },
            )
        except Exception:
            pass
        return {**payload, "events": []}

    def complete_session(
        self,
        user_id: str,
        session_id: str,
        summary: Optional[Dict[str, Any]] = None,
        *,
        ended_at: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        inferred_user = _user_from_session_id(session_id)
        if inferred_user and inferred_user != user_id:
            return None
        ended_at = ended_at or _now_iso()
        existing = self.get_session(session_id)
        if existing and existing.get("user_id") and existing.get("user_id") != user_id:
            return None
        subject_id = existing.get("subject_id") if existing else None
        session_type = existing.get("session_type") if existing else None
        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "summary": summary or {},
            "ended_at": ended_at,
            "subject_id": subject_id,
            "session_type": session_type,
        }
        self._log(user_id, "session_completed", payload)
        try:
            xapi.emit(
                user_id,
                "http://adlnet.gov/expapi/verbs/terminated",
                f"session:{session_id}",
                context={
                    "session_type": session_type,
                    "subject": subject_id,
                    "summary": payload["summary"],
                },
            )
        except Exception:
            pass
        if existing:
            existing.setdefault("summary", {}).update(payload["summary"])
            existing["ended_at"] = ended_at
            return existing
        return self.get_session(session_id)

    def record_event(
        self,
        user_id: str,
        subject_id: Optional[str],
        event_type: str,
        *,
        lesson_id: Optional[str] = None,
        score: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        skill_id: Optional[str] = None,
        competency_id: Optional[str] = None,
        outcome: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        recorded_at = _now_iso()
        enriched_details = enrich_event_metadata(
            details,
            subject_id=subject_id,
            skill_id=skill_id,
            competency_id=competency_id,
            score=score,
            event_type=event_type,
            outcome=outcome,
            status=status,
            graph=self._graph,
        )

        payload = {
            "event_id": uuid.uuid4().hex,
            "user_id": user_id,
            "subject_id": subject_id,
            "event_type": event_type,
            "lesson_id": lesson_id,
            "score": score,
            "details": enriched_details,
            "session_id": session_id,
            "recorded_at": recorded_at,
        }
        skill_meta = (
            enriched_details.get("skill_id")
            or enriched_details.get("competency_id")
            or skill_id
            or competency_id
        )
        if skill_meta:
            payload["skill_id"] = skill_meta
            payload.setdefault("competency_id", enriched_details.get("competency_id") or skill_meta)
            if not payload.get("subject_id"):
                payload["subject_id"] = skill_meta

        outcome_meta = enriched_details.get("outcome")
        if outcome_meta:
            payload["outcome"] = outcome_meta
        status_meta = enriched_details.get("status")
        if status_meta:
            payload["status"] = status_meta

        op = "session_event" if session_id else "standalone_event"
        self._log(user_id, op, payload)
        try:
            xapi.emit(
                user_id,
                "http://adlnet.gov/expapi/verbs/experienced",
                f"event:{payload['event_id']}",
                score=score,
                context={
                    "event_type": event_type,
                    "subject": subject_id,
                    "lesson_id": lesson_id,
                    "session_id": session_id,
                },
            )
        except Exception:
            pass
        return payload

    # ------------------------------------------------------------------
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        user_id = _user_from_session_id(session_id)
        if not user_id:
            return None
        entries = self._db.list_journey(user_id=user_id, limit=self._history_window)
        session = _build_sessions(entries).get(session_id)
        if session:
            session.setdefault("summary", {})
            session.setdefault("events", [])
        return session

    def get_timeline(
        self,
        user_id: str,
        *,
        subject_id: Optional[str] = None,
        limit_sessions: int = 10,
        limit_events: int = 50,
    ) -> Dict[str, Any]:
        entries = self._db.list_journey(user_id=user_id, limit=self._history_window)
        sessions = _build_sessions(entries)
        filtered_sessions: List[Dict[str, Any]] = []
        for session in sessions.values():
            if subject_id and session.get("subject_id") != subject_id:
                continue
            filtered_sessions.append(session)
        filtered_sessions.sort(key=lambda s: s.get("started_at") or "", reverse=True)
        if limit_sessions:
            filtered_sessions = filtered_sessions[: max(0, int(limit_sessions))]
        kept_session_ids = {s.get("session_id") for s in filtered_sessions}

        loose_events: List[Dict[str, Any]] = []
        for entry in entries:
            payload = entry.get("payload") or {}
            op = entry.get("op")
            if op not in {"standalone_event", "session_event", "learning_event_recorded"}:
                continue
            if subject_id and payload.get("subject_id") != subject_id:
                continue
            sid = payload.get("session_id")
            if sid and sid in kept_session_ids:
                continue
            loose_event = {
                "event_type": payload.get("event_type"),
                "details": payload.get("details", {}),
                "recorded_at": payload.get("recorded_at") or entry.get("created_at"),
                "subject_id": payload.get("subject_id"),
                "score": payload.get("score"),
                "session_id": sid,
            }
            loose_events.append(loose_event)
        reflection_events: List[Dict[str, Any]] = []
        try:
            recent_assessments = self._db.list_recent_assessments(
                user_id,
                topic=subject_id,
                limit=max(int(limit_events), 1) if limit_events else 50,
            )
        except Exception:
            recent_assessments = []
        for attempt in recent_assessments:
            reflection_text = attempt.get("self_assessment")
            if not reflection_text:
                continue
            reflection_events.append(
                {
                    "event_type": "self_assessment",
                    "details": {
                        "self_assessment": reflection_text,
                        "item_id": attempt.get("item_id"),
                        "bloom_level": attempt.get("bloom_level"),
                        "confidence": attempt.get("confidence"),
                        "source": attempt.get("source"),
                    },
                    "recorded_at": attempt.get("created_at"),
                    "subject_id": attempt.get("domain"),
                    "score": attempt.get("score"),
                    "session_id": None,
                }
            )

        if reflection_events:
            loose_events.extend(reflection_events)

        def _event_order_key(event: Dict[str, Any]) -> str:
            value = event.get("recorded_at")
            if isinstance(value, str):
                return value
            return str(value or "")

        loose_events.sort(key=_event_order_key, reverse=True)
        if limit_events:
            loose_events = loose_events[: max(0, int(limit_events))]

        total_duration = 0.0
        aggregate_events: List[Dict[str, Any]] = []
        last_activity: Optional[datetime] = None
        for session in filtered_sessions:
            insights = _derive_session_insights(session)
            session["insights"] = insights
            duration = insights.get("duration_seconds")
            if isinstance(duration, (int, float)):
                total_duration += float(duration)
            _append_session_events(aggregate_events, session.get("events", []), session.get("session_id"))
            last_activity = _max_datetime(last_activity, _parse_iso_timestamp(insights.get("last_activity_at")))

        for event in loose_events:
            aggregate_events.append(
                {
                    "session_id": event.get("session_id"),
                    "event_type": event.get("event_type"),
                    "details": event.get("details", {}),
                    "recorded_at": event.get("recorded_at"),
                    "score": event.get("score"),
                }
            )
            last_activity = _max_datetime(last_activity, _parse_iso_timestamp(event.get("recorded_at")))

        aggregate_events.sort(key=_event_time_key)
        bloom_transitions: List[Dict[str, Any]] = []
        levels_seen: List[str] = []
        previous_level: Optional[str] = None
        running_success = 0
        running_failure = 0
        longest_success = 0
        total_failures = 0
        domain_skill_counts: Counter[str] = Counter()
        domain_media_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        raw_skill_counts: Counter[str] = Counter()
        for event in aggregate_events:
            details = event.get("details") or {}
            area = details.get("skill_area") or details.get("competency_id") or event.get("subject_id")
            if area:
                area_text = str(area)
                raw_skill_counts[area_text] += 1
                domain_key = _normalise_skill_area(area_text)
                domain_skill_counts[domain_key] += 1
                media_channel = details.get("media_channel")
                if media_channel:
                    domain_media_counts[domain_key][str(media_channel)] += 1
            level = _extract_bloom_level(event)
            if level:
                if level not in levels_seen:
                    levels_seen.append(level)
                if previous_level and level != previous_level:
                    bloom_transitions.append(
                        {
                            "from": previous_level,
                            "to": level,
                            "at": event.get("recorded_at"),
                            "session_id": event.get("session_id"),
                        }
                    )
                previous_level = level
            outcome = _event_success(event)
            if outcome is True:
                running_success += 1
                running_failure = 0
                if running_success > longest_success:
                    longest_success = running_success
            elif outcome is False:
                total_failures += 1
                running_failure += 1
                running_success = 0
            else:
                running_success = 0
                running_failure = 0

        bpmn_feedback = _derive_bpmn_feedback(aggregate_events)

        domain_breakdown = {}
        for domain, count in domain_skill_counts.items():
            media_counts = domain_media_counts.get(domain, Counter())
            domain_breakdown[domain] = {
                "events": count,
                "media_channels": dict(
                    sorted(media_counts.items(), key=lambda item: (-item[1], item[0]))
                ),
            }

        timeline_insights = {
            "total_duration_seconds": round(total_duration, 2) if total_duration else 0.0,
            "last_activity_at": last_activity.strftime(ISO_FORMAT) if last_activity else None,
            "bloom": {
                "transitions": bloom_transitions,
                "levels_encountered": levels_seen,
                "last_level": previous_level,
            },
            "streaks": {
                "current_success_streak": running_success,
                "current_failure_streak": running_failure,
                "longest_success_streak": longest_success,
                "total_failures": total_failures,
            },
            "skill_areas": {
                "domains": domain_breakdown,
                "raw": dict(sorted(raw_skill_counts.items(), key=lambda item: (-item[1], item[0]))),
            },
        }

        summary = {
            "total_sessions": len(filtered_sessions),
            "open_sessions": sum(1 for s in filtered_sessions if not s.get("ended_at")),
            "events_in_sessions": sum(len(s.get("events", [])) for s in filtered_sessions),
            "loose_events": len(loose_events),
            "total_duration_seconds": timeline_insights["total_duration_seconds"],
            "distinct_bloom_levels": levels_seen,
            "longest_success_streak": timeline_insights["streaks"]["longest_success_streak"],
            "current_success_streak": timeline_insights["streaks"]["current_success_streak"],
            "current_failure_streak": timeline_insights["streaks"]["current_failure_streak"],
            "last_activity_at": timeline_insights["last_activity_at"],
            "skill_area_domain_counts": {domain: data["events"] for domain, data in domain_breakdown.items()},
        }

        nudges: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        if last_activity and now - last_activity > timedelta(hours=_INACTIVITY_THRESHOLD_HOURS):
            hours_inactive = (now - last_activity).total_seconds() / 3600.0
            nudges.append(
                {
                    "code": "inactivity",
                    "message": "Learner has been inactive for a sustained period; consider a supportive check-in.",
                    "triggered_at": _now_iso(),
                    "last_activity_at": timeline_insights["last_activity_at"],
                    "hours_inactive": round(hours_inactive, 1),
                }
            )
        if timeline_insights["streaks"]["current_failure_streak"] >= 3:
            nudges.append(
                {
                    "code": "failure_streak",
                    "message": "Multiple consecutive struggles detected; recommend remediation guidance.",
                    "triggered_at": _now_iso(),
                    "current_failure_streak": timeline_insights["streaks"]["current_failure_streak"],
                }
            )

        return {
            "sessions": filtered_sessions,
            "loose_events": loose_events,
            "summary": summary,
            "insights": timeline_insights,
            "nudges": nudges,
            "feedback": {"bpmn_modeling": bpmn_feedback},
        }

    # ------------------------------------------------------------------
    def _log(self, user_id: str, op: str, payload: Dict[str, Any]) -> None:
        try:
            self._db.log_journey_update(user_id, op, payload)
        except Exception:
            pass


def _build_sessions(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    ordered = list(reversed(entries))
    sessions: Dict[str, Dict[str, Any]] = {}
    for entry in ordered:
        payload = entry.get("payload") or {}
        op = entry.get("op")
        if op == "session_started":
            sid = payload.get("session_id")
            if not sid:
                continue
            session = sessions.setdefault(sid, {
                "session_id": sid,
                "user_id": payload.get("user_id"),
                "subject_id": payload.get("subject_id"),
                "session_type": payload.get("session_type"),
                "metadata": _ensure_metadata(payload.get("metadata")),
                "started_at": payload.get("started_at") or entry.get("created_at"),
                "events": [],
            })
            session.setdefault("metadata", _ensure_metadata(payload.get("metadata")))
            if not session.get("started_at"):
                session["started_at"] = payload.get("started_at") or entry.get("created_at")
            if payload.get("subject_id") and not session.get("subject_id"):
                session["subject_id"] = payload.get("subject_id")
            if payload.get("session_type") and not session.get("session_type"):
                session["session_type"] = payload.get("session_type")
        elif op in {"session_event", "learning_event_recorded"}:
            sid = payload.get("session_id")
            if not sid:
                continue
            session = sessions.setdefault(sid, {
                "session_id": sid,
                "user_id": payload.get("user_id"),
                "subject_id": payload.get("subject_id"),
                "session_type": None,
                "metadata": {},
                "started_at": None,
                "events": [],
            })
            event = {
                "event_type": payload.get("event_type"),
                "lesson_id": payload.get("lesson_id"),
                "details": payload.get("details", {}),
                "score": payload.get("score"),
                "recorded_at": payload.get("recorded_at") or entry.get("created_at"),
            }
            session.setdefault("events", []).append(event)
        elif op == "session_completed":
            sid = payload.get("session_id")
            if not sid:
                continue
            session = sessions.setdefault(sid, {
                "session_id": sid,
                "user_id": payload.get("user_id"),
                "subject_id": payload.get("subject_id"),
                "session_type": payload.get("session_type"),
                "metadata": {},
                "started_at": None,
                "events": [],
            })
            session["ended_at"] = payload.get("ended_at") or entry.get("created_at")
            session["summary"] = payload.get("summary", {})
            if payload.get("subject_id") and not session.get("subject_id"):
                session["subject_id"] = payload.get("subject_id")
            if payload.get("session_type") and not session.get("session_type"):
                session["session_type"] = payload.get("session_type")
    return sessions


__all__ = ["LearningJourneyTracker", "enrich_event_metadata"]
