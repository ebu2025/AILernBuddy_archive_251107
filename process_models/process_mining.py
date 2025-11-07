"""Utilities for lightweight process mining on adaptive learning event logs.

The module keeps dependencies minimal so that logs captured from the
learning journeys can be inspected inside notebook experiments or FastAPI
routes without introducing a full process mining suite.  It focuses on
metrics the orchestration team frequently requests: cycle time, variant
analysis, bottleneck identification, and dropout detection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from collections import defaultdict, Counter

from .process_utils import with_retry, with_timeout, ProcessExecutionError

logger = logging.getLogger(__name__)

__all__ = [
    "Event",
    "parse_event_log",
    "group_events_by_case",
    "calculate_cycle_times",
    "discover_variants",
    "identify_bottlenecks",
    "generate_process_diagnostics",
]


_AREA_ALIASES = {
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


@dataclass
class Event:
    """Simple representation of a learning journey event."""

    case_id: str
    activity: str
    timestamp: datetime
    skill_area: Optional[str] = None
    status: str = "complete"
    media_channel: Optional[str] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Event":
        """Create an :class:`Event` from a dictionary payload.

        The helper normalises timestamp strings and ensures metadata is a
        mutable mapping so downstream analysis can enrich events in-memory.
        """

        timestamp = _coerce_timestamp(payload.get("timestamp"))
        metadata = dict(payload.get("metadata", {}))
        skill_area = payload.get("skill_area") or metadata.get("skill_area")
        status = payload.get("status") or metadata.get("status", "complete")
        media_channel = payload.get("media_channel") or metadata.get("media_channel")
        return cls(
            case_id=str(payload["case_id"]),
            activity=str(payload["activity"]),
            timestamp=timestamp,
            skill_area=skill_area,
            status=status,
            media_channel=media_channel,
            metadata=metadata,
        )


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # assume UNIX timestamp seconds
        return datetime.fromtimestamp(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported timestamp format: {value}") from exc
    raise TypeError(f"Timestamp of type {type(value)!r} is not supported")


def _normalise_area(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    base = str(value)
    candidate = base.split(":", 1)[0]
    if "." in candidate:
        candidate = candidate.split(".", 1)[0]
    return _AREA_ALIASES.get(candidate.lower(), candidate.lower())


def parse_event_log(raw_events: Iterable[Mapping[str, Any]]) -> List[Event]:
    """Parse and chronologically sort raw event payloads."""

    events = [Event.from_dict(evt) for evt in raw_events]
    events.sort(key=lambda evt: (evt.case_id, evt.timestamp))
    return events


def group_events_by_case(events: Sequence[Event]) -> Dict[str, List[Event]]:
    """Group events by case identifier while maintaining chronological order."""

    grouped: Dict[str, List[Event]] = defaultdict(list)
    for event in events:
        grouped[event.case_id].append(event)
    # Ensure each case timeline is sorted even if raw input interleaves events
    for case_id in grouped:
        grouped[case_id].sort(key=lambda evt: evt.timestamp)
    return dict(grouped)


def calculate_cycle_times(events: Sequence[Event]) -> Dict[str, float]:
    """Return per-case cycle time in hours."""

    grouped = group_events_by_case(events)
    cycle_times: Dict[str, float] = {}
    for case_id, case_events in grouped.items():
        if len(case_events) == 1:
            cycle_times[case_id] = 0.0
            continue
        duration = case_events[-1].timestamp - case_events[0].timestamp
        cycle_times[case_id] = duration.total_seconds() / 3600.0
    return cycle_times


def discover_variants(events: Sequence[Event]) -> Dict[str, int]:
    """Discover process variants based on ordered activity sequences per case."""

    grouped = group_events_by_case(events)
    variants: Counter[str] = Counter()
    for case_events in grouped.values():
        variant = " > ".join(evt.activity for evt in case_events)
        variants[variant] += 1
    return dict(variants)


def identify_bottlenecks(events: Sequence[Event], top_n: Optional[int] = 3) -> List[Tuple[str, float]]:
    """Identify activities with the largest average wait to the next step."""

    grouped = group_events_by_case(events)
    waits: Dict[str, List[float]] = defaultdict(list)
    for case_events in grouped.values():
        for current, nxt in zip(case_events, case_events[1:]):
            delta_hours = (nxt.timestamp - current.timestamp).total_seconds() / 3600.0
            waits[current.activity].append(delta_hours)
    averages = [
        (activity, mean(values))
        for activity, values in waits.items()
        if values
    ]
    averages.sort(key=lambda item: item[1], reverse=True)
    if top_n is None:
        return averages
    return averages[:top_n]


def _aggregate_skill_area_cycle_times(
    events: Sequence[Event],
    cycle_times: Mapping[str, float],
) -> Dict[str, Dict[str, float]]:
    skill_area_cycles: Dict[str, List[float]] = defaultdict(list)
    grouped = group_events_by_case(events)
    for case_id, case_events in grouped.items():
        areas = {_normalise_area(evt.skill_area) for evt in case_events if evt.skill_area}
        if not areas:
            continue
        duration = cycle_times.get(case_id)
        if duration is None:
            continue
        for area in areas:
            skill_area_cycles[area].append(duration)
    summary: Dict[str, Dict[str, float]] = {}
    for area, durations in skill_area_cycles.items():
        summary[area] = {
            "cases": float(len(durations)),
            "mean_cycle_hours": mean(durations),
            "median_cycle_hours": median(durations),
        }
    return summary


def generate_process_diagnostics(
    events: Sequence[Event],
    *,
    end_activity: Optional[str] = None,
    top_n_bottlenecks: int = 3,
) -> Dict[str, Any]:
    """Return process diagnostics suitable for lightweight reporting."""

    if not events:
        return {
            "cases": 0,
            "cycle_time_hours": {"mean": 0.0, "median": 0.0},
            "variants": [],
            "bottlenecks": [],
            "dropouts": {"cases": [], "rate": 0.0},
            "skill_areas": {},
            "media_channels": {},
        }

    media_usage: Dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        area_key = _normalise_area(event.skill_area)
        if event.media_channel:
            media_usage[area_key][str(event.media_channel)] += 1

    cycle_times = calculate_cycle_times(events)
    cycle_duration_values = list(cycle_times.values())
    variants = discover_variants(events)
    variant_summary = [
        {
            "variant": variant.split(" > "),
            "count": count,
        }
        for variant, count in sorted(variants.items(), key=lambda item: item[1], reverse=True)
    ]
    bottlenecks = identify_bottlenecks(events, top_n=top_n_bottlenecks)

    grouped = group_events_by_case(events)
    dropout_cases: List[str] = []
    if end_activity:
        for case_id, case_events in grouped.items():
            if case_events[-1].activity != end_activity:
                dropout_cases.append(case_id)
    dropout_rate = (len(dropout_cases) / len(grouped)) if grouped else 0.0

    skill_area_summary = _aggregate_skill_area_cycle_times(events, cycle_times)
    for area, data in skill_area_summary.items():
        channels = media_usage.get(area, Counter())
        data["media_channels"] = dict(sorted(channels.items(), key=lambda item: (-item[1], item[0])))

    diagnostics = {
        "cases": len(grouped),
        "cycle_time_hours": {
            "mean": mean(cycle_duration_values) if cycle_duration_values else 0.0,
            "median": median(cycle_duration_values) if cycle_duration_values else 0.0,
        },
        "variants": variant_summary,
        "bottlenecks": [
            {"activity": activity, "mean_wait_hours": wait}
            for activity, wait in bottlenecks
        ],
        "dropouts": {"cases": dropout_cases, "rate": dropout_rate},
        "skill_areas": skill_area_summary,
        "media_channels": {
            area: data["media_channels"]
            for area, data in skill_area_summary.items()
            if data.get("media_channels")
        },
    }
    return diagnostics
