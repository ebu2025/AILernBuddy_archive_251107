import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter
from functools import lru_cache
from time import perf_counter
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import db
from bloom_levels import BLOOM_LEVELS
from prompts.masterprompts import MasterPrompt, get_prompt, load_prompts
from engines import progression
from engines.validation import (
    validate_assessment_result,
    validate_microcheck_data,
    validate_assessment_step,
    AssessmentValidationError
)

logger = logging.getLogger(__name__)

try:
    _K_LEVEL_SEQUENCE = BLOOM_LEVELS.k_level_sequence()
except Exception:
    _K_LEVEL_SEQUENCE = ("K1", "K2", "K3")

LOWEST_BLOOM_LEVEL = BLOOM_LEVELS.lowest_level()
DEFAULT_MICROCHECK_HINT = (
    _K_LEVEL_SEQUENCE[1] if len(_K_LEVEL_SEQUENCE) >= 2 else LOWEST_BLOOM_LEVEL
)

# --------- Model/endpoint from environment ---------
MODEL_ID = os.getenv("MODEL_ID", "DeepSeek-R1-Distill-Qwen-14B")
GPT4ALL_URL = os.getenv("GPT4ALL_URL", "http://localhost:4891/v1/chat/completions")

# --------- Master prompt management ---------
_PROMPT_VARIANTS = load_prompts()
_DEFAULT_VARIANT = os.getenv("PROMPT_VARIANT", "socratic")
try:
    ACTIVE_MASTER_PROMPT: MasterPrompt = get_prompt(_DEFAULT_VARIANT)
except KeyError:
    ACTIVE_MASTER_PROMPT = get_prompt(next(iter(_PROMPT_VARIANTS)))

PROMPT_VARIANT = ACTIVE_MASTER_PROMPT.normalized_variant
PROMPT_VERSION = ACTIVE_MASTER_PROMPT.prompt_version
MASTER_PROMPTS = _PROMPT_VARIANTS

# --------- Universal system prompt (domain-neutral) ---------
SUPPORTED_THEMES = {
    "bpmn": "BPMN 2.0 process modelling",
    "business_process": "Business process management",
    "language": "Language and communication skills",
    "language_de_en": "German↔English bilingual competence",
    "language_zh_en": "Chinese↔English bilingual competence",
    "mathematics": "Mathematical problem-solving competence",
}

SYSTEM_TUTOR_TEMPLATE = ACTIVE_MASTER_PROMPT.system_template


# The tutor may optionally append structured database operations at the end of a reply.
# Our app parses this block and applies the relevant actions.
SYSTEM_TUTOR_JSON = """The JSON block at the end of your reply may include database operations.
Only the operations below are supported and the learner's language/topic must stay consistent:

{
  "db_ops": [
    {"op": "add_prompt", "payload": {"topic": "<CURRENT_TOPIC>", "prompt_text": "<ONLY in the learner's language>"}},
    {"op": "log_rationale", "payload": {"note": "<why the next activity is appropriate>"}},
    {"op": "suggest_next_item", "payload": {"skill": "<skill_id>", "reason": "<short justification>"}}
  ]
}

If you have nothing to log, set "db_ops": []. Do not invent other tables or languages.
Beende deine Antwort immer mit genau einem gültigen JSON-Block (Objekt, kein Array, keine Code-Fences) und füge danach keinen weiteren Text hinzu."""


def _level_label(level: str) -> str:
    entry = BLOOM_LEVELS.get(level)
    if entry:
        return f"{entry.id} – {entry.label}"
    return f"{level} – learning status"


def _bloom_reference() -> str:
    return BLOOM_LEVELS.formatted_overview()


BLOOM_LEVEL_REFERENCE = _bloom_reference()
_LEVEL_SEQUENCE = BLOOM_LEVELS.sequence()
BLOOM_LEVEL_SEQUENCE_TEXT = ", ".join(_LEVEL_SEQUENCE) if _LEVEL_SEQUENCE else "<no levels configured>"

JSON_RESPONSE_DIRECTIVE = ACTIVE_MASTER_PROMPT.json_instructions.format(
    BLOOM_LEVEL_SEQUENCE_TEXT=BLOOM_LEVEL_SEQUENCE_TEXT,
    BLOOM_LEVEL_REFERENCE=BLOOM_LEVEL_REFERENCE,
)
JSON_RESPONSE_DIRECTIVE += (
    "\nUse the schema above exactly. The `diagnosis` value must be one of "
    "['conceptual', 'procedural', 'careless', 'none'] (treat 'none' as no "
    "detected misconception). Clamp `assessment.score`, "
    "`assessment.confidence`, and `microcheck_score` to the [0, 1] range."
    "\nBeende deine Antwort immer mit genau einem gültigen JSON-Block (Objekt, kein Array, keine Markdown-Fences) "
    "als letztem Abschnitt – keine weiteren Inhalte danach."
)


# Cache configuration for the system prompt builder
_BUILD_SYSTEM_PROMPT_CACHE_TTL_SECONDS = 5 * 60
_build_system_prompt_cache_lock = threading.Lock()
_build_system_prompt_cache_last_cleared = time.monotonic()


def _normalize_struggles(struggles: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not struggles:
        return ()
    if isinstance(struggles, str):
        cleaned = struggles.strip()
        return (cleaned,) if cleaned else ()

    normalized: list[str] = []
    for entry in struggles:
        if entry is None:
            continue
        text = str(entry).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


@lru_cache(maxsize=256)
def _build_system_prompt_cached(
    theme_key_lower: str,
    theme_fallback_title: str,
    level: str,
    struggles_tuple: Tuple[str, ...],
) -> str:
    theme = SUPPORTED_THEMES.get(theme_key_lower, theme_fallback_title)
    level_label = _level_label(level)
    struggle_text = _format_struggles(struggles_tuple)
    return SYSTEM_TUTOR_TEMPLATE.format(
        theme=theme,
        level_label=level_label,
        struggles=struggle_text,
    )


# --------- System-Prompt Builder ---------
def build_system_prompt(
    theme_key: str,
    level: str,
    struggles: Optional[Sequence[str]] = None,
) -> str:
    """Compose the master system prompt for the local GPT tutor."""

    normalized_struggles = _normalize_struggles(struggles)
    theme_key_lower = theme_key.lower()
    fallback_theme = theme_key.title()
    now = time.monotonic()

    with _build_system_prompt_cache_lock:
        global _build_system_prompt_cache_last_cleared
        if now - _build_system_prompt_cache_last_cleared >= _BUILD_SYSTEM_PROMPT_CACHE_TTL_SECONDS:
            _build_system_prompt_cached.cache_clear()
            _build_system_prompt_cache_last_cleared = now
            logger.debug("build_system_prompt cache cleared after TTL expiry")

        prompt = _build_system_prompt_cached(
            theme_key_lower,
            fallback_theme,
            level,
            normalized_struggles,
        )
        cache_info = _build_system_prompt_cached.cache_info()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "build_system_prompt cache stats: hits=%d, misses=%d, current_size=%d",
            cache_info.hits,
            cache_info.misses,
            cache_info.currsize,
        )

    return prompt


def _format_struggles(struggles: Optional[Sequence[str]]) -> str:
    if not struggles:
        return "No specific obstacles reported."
    cleaned: Iterable[str] = (s.strip() for s in struggles if s and s.strip())
    bullet_points = [f"• {entry}" for entry in cleaned if entry]
    return "\n  " + "\n  ".join(bullet_points) if bullet_points else "No specific obstacles reported."


# Backwards compatibility for legacy callers
SYSTEM_TUTOR = build_system_prompt("bpmn", LOWEST_BLOOM_LEVEL, [])


# --------- Small Item Bank (diagnostic/practice examples; cross-domain) ----------
# Every item definition MUST include a 'body' that is written into the database.
ITEM_BANK = {
    # BPMN / process modelling
    "bpmn_swimlanes": {
        "domain": "bpmn",
        "topic": "BPMN Swimlanes and Pools",
        "skill_id": "bpmn.swimlanes",
        "tags": ["bpmn", "process-modelling", "swimlanes", "responsibility"],
        "items": [
            {
                "id": "bpmn_swim_01",
                "skill_id": "bpmn.swimlanes.concepts",
                "bloom_level": "K1",
                "difficulty": -0.2,
                "elo_target": 980.0,
                "answer_key": "Swimlanes partition BPMN diagrams by participant or role to make responsibilities visible.",
                "diagnosis_focus": "conceptual",
                "tags": ["definition", "roles"],
                "body": "Explain in 2–3 sentences what BPMN swimlanes are used for and what they visualise.",
            },
            {
                "id": "bpmn_swim_02",
                "skill_id": "bpmn.swimlanes.application",
                "bloom_level": "K2",
                "difficulty": 0.0,
                "elo_target": 1000.0,
                "answer_key": "Scenario mentions two pools (customer, shop) with internal lanes and events/tasks linked by sequence flow plus a message flow between pools.",
                "diagnosis_focus": "procedural",
                "tags": ["scenario", "message-flow"],
                "body": "Provide a short text example with two pools (customer, shop) and one to two lanes each. Describe an order process in words (start/end event, tasks, message flow).",
            },
            {
                "id": "bpmn_swim_03",
                "skill_id": "bpmn.swimlanes.communication",
                "bloom_level": "K3",
                "difficulty": 0.3,
                "elo_target": 1030.0,
                "answer_key": "Sequence flow stays inside a pool showing task order, while message flow connects different pools to show information exchange.",
                "diagnosis_focus": "conceptual",
                "tags": ["comparison", "flows"],
                "body": "State the difference between sequence flow and message flow in BPMN in one to two sentences each.",
            },
        ],
    },
    # Mathematics / quadratic functions
    "mathe_quadratisch": {
        "domain": "math",
        "topic": "Quadratic Functions",
        "skill_id": "math.quadratic_functions",
        "tags": ["mathematics", "quadratic", "functions"],
        "items": [
            {
                "id": "math_quad_01",
                "skill_id": "math.quadratic.definition",
                "bloom_level": "K1",
                "difficulty": -0.3,
                "elo_target": 970.0,
                "answer_key": "A quadratic function has form f(x)=ax^2+bx+c with a≠0 and its graph is a parabola opening according to the sign of a.",
                "diagnosis_focus": "conceptual",
                "tags": ["definition", "parabola"],
                "body": "Define a quadratic function. Name the general form f(x)=ax^2+bx+c and describe the parabola in one to two sentences.",
            },
            {
                "id": "math_quad_02",
                "skill_id": "math.quadratic.procedural",
                "bloom_level": "K2",
                "difficulty": 0.0,
                "elo_target": 1000.0,
                "answer_key": "Zeros at x=1 and x=3 with vertex at (2,-1) after completing the square or using -b/(2a).",
                "diagnosis_focus": "procedural",
                "tags": ["worked-example", "roots"],
                "body": "Given f(x)=x^2-4x+3. Determine the zeros and the vertex (show the short working).",
            },
            {
                "id": "math_quad_03",
                "skill_id": "math.quadratic.analysis",
                "bloom_level": "K3",
                "difficulty": 0.4,
                "elo_target": 1040.0,
                "answer_key": "Parameter a controls opening: positive gives upward, negative downward, |a|>1 narrows (stretch), |a|<1 widens (compression).",
                "diagnosis_focus": "conceptual",
                "tags": ["parameter", "interpretation"],
                "body": "Explain in 3–4 sentences how the parameter a in f(x)=a x^2 + b x + c affects the opening and stretching/compression of the parabola.",
            },
        ],
    },
    # Business / NPV
    "wirtschaft_npv": {
        "domain": "business",
        "topic": "Net Present Value (NPV)",
        "skill_id": "business.npv",
        "tags": ["finance", "npv", "discounting"],
        "items": [
            {
                "id": "biz_npv_01",
                "skill_id": "business.npv.concept",
                "bloom_level": "K1",
                "difficulty": -0.1,
                "elo_target": 990.0,
                "answer_key": "NPV sums discounted future cash flows minus initial investment to express today's value using the discount rate.",
                "diagnosis_focus": "conceptual",
                "tags": ["definition", "discounting"],
                "body": "Define the net present value (NPV) in 2–3 sentences and describe the core idea of discounting.",
            },
            {
                "id": "biz_npv_02",
                "skill_id": "business.npv.procedural",
                "bloom_level": "K2",
                "difficulty": 0.1,
                "elo_target": 1010.0,
                "answer_key": "Discounted cash flows (600/1.05 + 500/1.05^2) ≈ 1029 - 1000 ≈ 29 CHF so the project adds small value.",
                "diagnosis_focus": "procedural",
                "tags": ["calculation", "example"],
                "body": "Short worked example: investment of 1,000 CHF today; cashflows of 600/500 CHF over the next two years; discount rate 5%. Estimate the NPV and interpret the result.",
            },
            {
                "id": "biz_npv_03",
                "skill_id": "business.npv.limits",
                "bloom_level": "K3",
                "difficulty": 0.35,
                "elo_target": 1035.0,
                "answer_key": "Limitations include relying on uncertain forecasts/discount rates and ignoring qualitative factors or liquidity constraints.",
                "diagnosis_focus": "conceptual",
                "tags": ["limitations", "evaluation"],
                "body": "Name two limitations of the NPV concept in one to two sentences each.",
            },
        ],
    },
    # General / writing & argumentation (cross-domain)
    "argumentieren_allgemein": {
        "domain": "writing",
        "topic": "Argumentation and Academic Writing",
        "skill_id": "writing.argumentation",
        "tags": ["writing", "argumentation", "communication"],
        "items": [
            {
                "id": "arg_allg_01",
                "skill_id": "writing.argumentation.debate",
                "bloom_level": "K3",
                "difficulty": -0.2,
                "elo_target": 980.0,
                "answer_key": "Response balances pro and contra reasons with a concluding judgement that weighs both perspectives.",
                "diagnosis_focus": "communication",
                "tags": ["debate", "balance"],
                "body": "Write a short argument (5–7 sentences) for and against the thesis: \"Homework should be abolished.\" Finish with a balanced conclusion.",
            },
            {
                "id": "arg_allg_02",
                "skill_id": "writing.argumentation.revision",
                "bloom_level": "K2",
                "difficulty": 0.2,
                "elo_target": 1020.0,
                "answer_key": "Improved version clarifies vague claims, corrects grammar, and justifies two concrete edits tied to clarity or accuracy.",
                "diagnosis_focus": "procedural",
                "tags": ["editing", "clarity"],
                "body": "Revise the following paragraph for clarity, precision, and correctness. Briefly justify two of your changes: \"Productivity is much better nowadays because tools do almost everything. As a result people work efficiently and feel less stressed.\"",
            },
            {
                "id": "arg_allg_03",
                "skill_id": "writing.argumentation.research",
                "bloom_level": "K3",
                "difficulty": 0.4,
                "elo_target": 1040.0,
                "answer_key": "Strong answers discuss scope, operationalisation, relevance, and feasibility and propose an example research question illustrating them.",
                "diagnosis_focus": "conceptual",
                "tags": ["research", "criteria"],
                "body": "Describe in 4–6 sentences the characteristics of a strong research question (scope, operationalisability, relevance, feasibility) and provide your own example.",
            },
        ],
    },
}


_LOGGER = logging.getLogger(__name__)


def _ensure_dict(value: Optional[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _json_log(event: str, payload: Dict[str, Any]) -> None:
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


def retry_question(
    user_id: str,
    subject_id: str,
    activity_id: Optional[str],
    *,
    current_target: float | int | None,
    diagnosis: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    db_module=db,
) -> dict[str, Any]:
    """Adjust the target ELO for a retry and persist the decision."""

    timer_start = perf_counter()
    try:
        target_before = float(current_target) if current_target is not None else 0.0
    except (TypeError, ValueError):
        target_before = 0.0

    result = progression.retry_question(target_before, diagnosis)
    target_after = float(result.get("target_elo", target_before) or 0.0)
    delta = float(result.get("delta", target_after - target_before) or 0.0)
    norm_diag = result.get("diagnosis")

    metadata_dict = _ensure_dict(metadata)

    log_payload = {
        "subject_id": subject_id,
        "activity_id": activity_id,
        "target_elo_before": target_before,
        "target_elo_after": target_after,
        "delta": delta,
        "diagnosis": norm_diag,
        "metadata": metadata_dict,
    }

    try:
        db_module.log_journey_update(user_id, "retry_question", log_payload)
    except Exception:
        _LOGGER.debug("Failed to log retry_question decision", exc_info=True)

    latency_seconds = metadata_dict.get("latency_seconds")
    if latency_seconds is None:
        latency_ms = metadata_dict.get("latency_ms")
        if latency_ms is not None:
            try:
                latency_seconds = float(latency_ms) / 1000.0
            except (TypeError, ValueError):
                latency_seconds = None

    confidence_signal = metadata_dict.get("confidence")
    if confidence_signal is None:
        confidence_signal = metadata_dict.get("normalized_confidence")

    bloom_before = (
        metadata_dict.get("bloom_level_before")
        or metadata_dict.get("bloom_level")
        or metadata_dict.get("current_bloom_level")
    )
    bloom_after = metadata_dict.get("bloom_level_after") or bloom_before

    reason = metadata_dict.get("reason")
    if not reason:
        diagnosis_label = norm_diag or "default"
        reason = f"Retry target adjusted after {diagnosis_label} diagnosis"

    processing_ms = round((perf_counter() - timer_start) * 1000.0, 3)

    json_payload: Dict[str, Any] = {
        "user_id": user_id,
        "subject_id": subject_id,
        "activity_id": activity_id,
        "prompt_variant": metadata_dict.get("prompt_variant", PROMPT_VARIANT),
        "bloom_level_before": bloom_before,
        "bloom_level_after": bloom_after,
        "elo_before": target_before,
        "elo_after": target_after,
        "confidence": confidence_signal,
        "decision": "retry_question",
        "reason": reason,
        "diagnosis": norm_diag,
        "latency_seconds": latency_seconds,
        "processing_ms": processing_ms,
        "delta": delta,
    }
    _json_log("retry_question_decision", json_payload)

    return {
        "user_id": user_id,
        "subject_id": subject_id,
        "activity_id": activity_id,
        "target_elo": target_after,
        "delta": delta,
        "diagnosis": norm_diag,
    }

def build_prompt_for_item(item: dict) -> str:
    """
    Build a suitable prompt for the LLM from the item body.
    """
    body = (item or {}).get("body", "").strip()
    return f"Task: {body}\nPlease answer concisely, accurately, and transparently."

# --------- Seed: write items into DB (with body) ----------
_SEED_LOCK = threading.Lock()
_ITEMS_SEEDED = False


def ensure_seed_items() -> None:
    """Populate the item bank in the database exactly once."""

    global _ITEMS_SEEDED

    if _ITEMS_SEEDED:
        return

    with _SEED_LOCK:
        if _ITEMS_SEEDED:
            return

        # Make sure the schema is present before we upsert anything.
        db.init()

        def _normalize_tags(raw: Any) -> list[str]:
            if not raw:
                return []
            if isinstance(raw, str):
                value = raw.strip()
                return [value] if value else []
            tags: list[str] = []
            try:
                iterator = iter(raw)
            except TypeError:
                text = str(raw).strip()
                return [text] if text else []
            for entry in iterator:
                if entry is None:
                    continue
                text = str(entry).strip()
                if text:
                    tags.append(text)
            return tags

        bank_entries: list[dict[str, Any]] = []

        for skill_key, spec in ITEM_BANK.items():
            spec_dict = spec or {}
            domain = str(spec_dict.get("domain") or skill_key).split("_")[0].lower()
            default_skill = spec_dict.get("skill_id") or skill_key

            base_metadata = dict(_ensure_dict(spec_dict.get("metadata")))
            topic = spec_dict.get("topic")
            if topic and "topic" not in base_metadata:
                base_metadata["topic"] = topic

            base_tags = _normalize_tags(spec_dict.get("tags"))
            if base_tags and "tags" not in base_metadata:
                base_metadata["tags"] = base_tags

            default_bloom = spec_dict.get("default_bloom_level") or LOWEST_BLOOM_LEVEL

            for it in spec_dict.get("items", []):
                it_id = it["id"]
                diff = float(it.get("difficulty", 0.0))
                body = it.get("body", "").strip() or f"Short practice for {skill_key} (item {it_id})."
                skill_value = it.get("skill_id") or default_skill
                db.upsert_item(it_id, skill_value, diff, body)

                bloom_level = str(it.get("bloom_level") or default_bloom)
                try:
                    elo_target = float(it.get("elo_target", 0.0))
                except (TypeError, ValueError):
                    elo_target = 0.0

                metadata = dict(base_metadata)
                item_metadata = _ensure_dict(it.get("metadata"))
                if item_metadata:
                    metadata.update(item_metadata)

                item_topic = it.get("topic")
                if item_topic and "topic" not in metadata:
                    metadata["topic"] = str(item_topic)

                tag_sources = []
                tag_sources.extend(_normalize_tags(base_metadata.get("tags")))
                tag_sources.extend(_normalize_tags(it.get("tags")))
                tag_sources.extend(_normalize_tags(item_metadata.get("tags")))
                if tag_sources:
                    seen_tags: set[str] = set()
                    unique_tags: list[str] = []
                    for tag in tag_sources:
                        lowered = tag.lower()
                        if lowered not in seen_tags:
                            seen_tags.add(lowered)
                            unique_tags.append(tag)
                    metadata["tags"] = unique_tags
                elif "tags" in metadata and not metadata.get("tags"):
                    metadata.pop("tags", None)

                diagnosis_focus = it.get("diagnosis_focus")
                if diagnosis_focus:
                    metadata["diagnosis_focus"] = str(diagnosis_focus)

                references = it.get("references")
                if references is not None and not isinstance(references, list):
                    references = [references]

                bank_entries.append(
                    {
                        "id": it_id,
                        "domain": domain,
                        "skill_id": skill_value,
                        "bloom_level": bloom_level,
                        "stimulus": body,
                        "elo_target": elo_target,
                        "answer_key": it.get("answer_key"),
                        "rubric_id": it.get("rubric_id"),
                        "difficulty": diff,
                        "metadata": metadata or None,
                        "references": references,
                        "exposure_limit": it.get("exposure_limit"),
                    }
                )

        if bank_entries:
            db.upsert_item_bank_entries(bank_entries)

        _ITEMS_SEEDED = True


def _normalize_bloom_id(level: Optional[str]) -> Optional[str]:
    if not level:
        return None
    candidate = str(level).strip().upper()
    try:
        BLOOM_LEVELS.index(candidate)
    except Exception:
        return None
    return candidate


def _microcheck_domain(topic_lower: str) -> str:
    if "bpmn" in topic_lower or "process" in topic_lower:
        return "bpmn"
    if "language" in topic_lower or "lingu" in topic_lower:
        return "language"
    if "math" in topic_lower or "algebra" in topic_lower:
        return "math"
    return "generic"


_BLOOM_TIERS: dict[str, str] = {}
for level_id in BLOOM_LEVELS.sequence():
    try:
        idx = BLOOM_LEVELS.index(level_id)
    except ValueError:
        idx = 0
    if idx <= 1:
        tier = "recall"
    elif idx == 2:
        tier = "apply"
    else:
        tier = "analyze"
    _BLOOM_TIERS[level_id] = tier


_RECENT_FOCUS_KEYWORDS: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "bpmn": (
        (("gateway", "xor", "exclusive"), "exclusive gateway usage"),
        (("parallel", "and", "simultaneous"), "parallel coordination"),
        (("event", "message", "signal"), "event handling"),
    ),
    "language": (
        (("translate", "translation"), "translation accuracy"),
        (("grammar", "tense", "syntax"), "grammar control"),
        (("vocabulary", "word", "term"), "vocabulary choice"),
    ),
    "math": (
        (("equation", "solve"), "equation solving"),
        (("proof", "justify"), "proof reasoning"),
    ),
}


def _recent_focus(domain: str, recent_answer: Optional[str]) -> Optional[str]:
    if not recent_answer:
        return None
    lowered = recent_answer.lower()
    for keywords, label in _RECENT_FOCUS_KEYWORDS.get(domain, ()):  # type: ignore[arg-type]
        if any(keyword in lowered for keyword in keywords):
            return label
    return None


def _inject_long_term_hint(base_hint: str, focus: Optional[str]) -> str:
    if not focus:
        return base_hint
    focus_text = focus.strip()
    if not focus_text:
        return base_hint
    lower_focus = focus_text.lower()
    if lower_focus in (base_hint or "").lower():
        return base_hint
    if base_hint.endswith("."):
        return f"{base_hint} Langfristiges Ziel: {focus_text}."
    return f"{base_hint}. Langfristiges Ziel: {focus_text}."


def _inject_long_term_question(question: str, focus: Optional[str]) -> str:
    if not focus:
        return question
    focus_text = focus.strip()
    if not focus_text:
        return question
    lower_focus = focus_text.lower()
    if lower_focus in question.lower():
        return question
    return f"{question} (Langfristiges Ziel: {focus_text})"


def _build_bpmn_microcheck(tier: str, hint: str, focus: Optional[str]) -> dict[str, Any]:
    if tier == "recall":
        return {
            "question": "In one sentence: when do you choose an exclusive (XOR) gateway in BPMN?",
            "answer_key": "Use it when exactly one outgoing path may continue.",
            "rubric": {
                "criteria": [
                    "mentions exclusivity or single-path control",
                    "contrasts it with parallel/AND flow or describes the condition",
                ]
            },
            "hint": hint,
        }
    if tier == "apply":
        if focus == "parallel coordination":
            question = (
                "A process must launch two review tasks at the same time before merging. "
                "Which BPMN gateway best models this and what completes the join?"
            )
            answer_key = "parallel gateway that synchronises branches when both complete"
            rubric = {
                "criteria": [
                    "identifies the parallel/AND gateway",
                    "explains that completion waits for both branches",
                ]
            }
        else:
            question = (
                "A workflow branches into approval and rejection outcomes. "
                "Which BPMN gateway should you use and what condition lets the flow continue?"
            )
            answer_key = "exclusive gateway with conditions tied to each outcome"
            rubric = {
                "criteria": [
                    "selects an exclusive gateway",
                    "connects the branch condition to how the flow continues",
                ]
            }
        return {"question": question, "answer_key": answer_key, "rubric": rubric, "hint": hint}
    # analyze tier and above
    focus_text = focus or "the gateway choice"
    return {
        "question": (
            f"Your recent answer highlighted {focus_text}. Describe one risk if the wrong gateway is used "
            "and how you would detect it early."
        ),
        "answer_key": "Explain a misrouting or deadlock risk and name a detection signal",
        "rubric": {
            "criteria": [
                "states a concrete risk such as deadlock or skipped path",
                "names an indicator or check that would reveal the issue",
            ]
        },
        "hint": hint,
    }


def _build_language_microcheck(tier: str, hint: str, focus: Optional[str]) -> dict[str, Any]:
    if tier == "recall":
        return {
            "question": "Give a one-sentence definition of the key term we just discussed.",
            "answer_key": "definition of the term",
            "rubric": {"criteria": ["states the meaning", "mentions the relevant context"]},
            "hint": hint,
        }
    if tier == "apply":
        focus_text = focus or "the term"
        return {
            "question": f"Use {focus_text} correctly in a new sentence that fits our topic.",
            "answer_key": "contextual sentence using the term appropriately",
            "rubric": {
                "criteria": ["sentence fits the topic", "term is used with the right meaning"]
            },
            "hint": hint,
        }
    return {
        "question": (
            "Critique this example sentence for nuance and register: 'The proposal was immaculate amazing.' "
            "What would you change and why?"
        ),
        "answer_key": "identify register clash and suggest a better adjective",
        "rubric": {
            "criteria": [
                "spots the tonal or register conflict",
                "proposes a correction with justification",
            ]
        },
        "hint": hint,
    }


def _build_math_microcheck(tier: str, hint: str, focus: Optional[str]) -> dict[str, Any]:
    if tier == "recall":
        return {
            "question": "State the fundamental principle you would use to balance a linear equation.",
            "answer_key": "perform the same operation on both sides",
            "rubric": {"criteria": ["mentions inverse operations", "applies to both sides"]},
            "hint": hint,
        }
    if tier == "apply":
        return {
            "question": "Solve for x: 3x + 5 = 14 and explain your main step.",
            "answer_key": "x = 3",
            "rubric": {"criteria": ["subtract 5", "divide by 3"]},
            "hint": hint,
        }
    focus_text = focus or "your previous reasoning"
    return {
        "question": f"Based on {focus_text}, explain how you would check if a derived formula holds for x = 0.",
        "answer_key": "substitute x = 0 and verify both sides remain equal",
        "rubric": {
            "criteria": ["substitution step", "states equality check or conclusion"]
        },
        "hint": hint,
    }


def _build_generic_microcheck(tier: str, hint: str, focus: Optional[str]) -> dict[str, Any]:
    if tier == "recall":
        return {
            "question": "State the core idea in one sentence from the concept we just covered.",
            "answer_key": "key idea",
            "rubric": {"criteria": ["identifies the main idea"]},
            "hint": hint,
        }
    if tier == "apply":
        return {
            "question": "Give a short example that shows how to apply the concept in practice.",
            "answer_key": "concrete example",
            "rubric": {"criteria": ["example fits concept", "explains outcome"]},
            "hint": hint,
        }
    focus_text = focus or "this concept"
    return {
        "question": f"Identify a limitation or risk when relying on {focus_text} and describe a safeguard.",
        "answer_key": "states limitation and mitigation",
        "rubric": {"criteria": ["names a risk", "offers a mitigation"]},
        "hint": hint,
    }


def generate_microcheck(
    topic: str,
    hint: Optional[str] = None,
    *,
    bloom_level: Optional[str] = None,
    recent_answer: Optional[str] = None,
    learning_focus: Optional[str] = None,
    learning_snapshot: Optional[Mapping[str, Any]] = None,
) -> dict:
    if not hint:
        hint = DEFAULT_MICROCHECK_HINT
    topic_lower = (topic or "").lower()
    domain = _microcheck_domain(topic_lower)
    normalized_level = _normalize_bloom_id(bloom_level)
    tier = _BLOOM_TIERS.get(normalized_level or "", "recall")
    focus = _recent_focus(domain, recent_answer)

    if domain == "bpmn":
        payload = _build_bpmn_microcheck(tier, hint, focus)
    elif domain == "language":
        payload = _build_language_microcheck(tier, hint, focus)
    elif domain == "math":
        payload = _build_math_microcheck(tier, hint, focus)
    else:
        payload = _build_generic_microcheck(tier, hint, focus)

    question_text = payload.get("question") or ""
    hint_text = payload.get("hint", hint)
    merged_hint = _inject_long_term_hint(hint_text, learning_focus)
    if merged_hint != hint_text:
        payload["hint"] = merged_hint
    merged_question = _inject_long_term_question(question_text, learning_focus)
    if merged_question != question_text:
        payload["question"] = merged_question

    payload.setdefault("metadata", {})
    if isinstance(payload["metadata"], dict):
        metadata: dict[str, Any] = payload["metadata"]
        if normalized_level:
            metadata.setdefault("bloom_level", normalized_level)
        if focus:
            metadata.setdefault("focus", focus)
        if learning_focus:
            metadata.setdefault("long_term_focus", learning_focus)
        if learning_snapshot:
            metadata.setdefault("learning_snapshot", dict(learning_snapshot))
    return payload



_RUBRIC_VALUE_KEYS = {
    "criteria",
    "expected",
    "keywords",
    "required",
    "must_include",
    "should_include",
}


def normalize_microcheck_rubric_terms(rubric: Any) -> list[str]:
    """Return a flat list of rubric terms usable for keyword checks."""

    def _iter_terms(value: Any) -> Iterable[str]:
        if value is None:
            return []
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        if isinstance(value, Mapping):
            terms: list[str] = []
            for key, nested in value.items():
                if key in _RUBRIC_VALUE_KEYS or nested is not None:
                    terms.extend(_iter_terms(nested))
            return terms
        if isinstance(value, Iterable):
            terms = []
            for entry in value:
                terms.extend(_iter_terms(entry))
            return terms
        return [str(value)]

    collected = []
    seen: set[str] = set()
    for term in _iter_terms(rubric):
        lowered = term.lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            collected.append(lowered)
    return collected


_TOKEN_PATTERN = re.compile(r"[\w']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _text_embedding(text: str) -> dict[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if not norm:
        return {}
    return {token: count / norm for token, count in counts.items()}


def _cosine_similarity(vec_a: Mapping[str, float], vec_b: Mapping[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(value * vec_b.get(key, 0.0) for key, value in vec_a.items())


def _score_from_similarity(similarity: float) -> float:
    if similarity >= 0.9:
        return 1.0
    if similarity >= 0.75:
        return 0.85
    if similarity >= 0.6:
        return 0.7
    if similarity >= 0.45:
        return 0.55
    if similarity >= 0.3:
        return 0.4
    if similarity >= 0.2:
        return 0.25
    return 0.0


def score_microcheck(reply: str, answer_key: str, rubric: Any) -> float:
    text = (reply or "").strip()
    if not text:
        return 0.0
    lower_text = text.lower()

    reference_parts: list[str] = []
    if answer_key:
        reference_parts.append(str(answer_key))
    if isinstance(rubric, Mapping):
        for key in ("expected", "criteria", "keywords"):
            value = rubric.get(key)
            if not value:
                continue
            if isinstance(value, str):
                reference_parts.append(value)
            elif isinstance(value, Iterable):
                reference_parts.append(" ".join(str(v) for v in value if v))
    elif isinstance(rubric, Iterable) and not isinstance(rubric, (str, bytes)):
        reference_parts.append(" ".join(str(entry) for entry in rubric if entry))

    reference_text = " ".join(reference_parts)
    similarity = _cosine_similarity(_text_embedding(lower_text), _text_embedding(reference_text.lower()))
    score = _score_from_similarity(similarity)

    reply_tokens = set(_tokenize(lower_text))
    answer_tokens = set(_tokenize(str(answer_key or "")))
    if reply_tokens and answer_tokens and reply_tokens.intersection(answer_tokens):
        score = max(score, 0.6)

    xor_markers = {"xor", "exclusive", "only", "exactly"}
    if any(marker in lower_text for marker in xor_markers):
        score = max(score, 0.6)

    rubric_terms = normalize_microcheck_rubric_terms(rubric)
    if rubric_terms:
        match_count = sum(1 for term in rubric_terms if term in lower_text)
        if match_count:
            if len(rubric_terms) == 1:
                fallback_score = 0.5
            else:
                coverage = match_count / len(rubric_terms)
                fallback_score = 0.3 + 0.4 * coverage
            score = max(score, fallback_score)

    return float(min(1.0, score))


def culture_sensitivity_check(text: str, locale: str) -> dict:
    sample_flags = ["idiom", "slang", "taboo"]
    lowered = (text or "").lower()
    flags = [flag for flag in sample_flags if flag in lowered]
    return {"locale": locale, "flags": flags}
