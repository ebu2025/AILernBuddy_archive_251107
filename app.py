# app.py — AILearnBuddy v4.4.6
# - Non-streaming (OpenAI-style params)
# - JSON/Code fence end detection to avoid false auto-continue
# - Minimal guards for db_ops.add_prompt (topic + language markers)

import asyncio
import copy
import math
import logging
import os, re, json, requests, hashlib, hmac, secrets, time
import statistics
import csv
import io
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from functools import lru_cache
from typing import Any, List, Optional, Literal, Sequence, Tuple, Mapping
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field, ValidationError

import db, tutor, journey, item_bank, rag, xapi
from bloom_levels import BLOOM_LEVELS

logger = logging.getLogger(__name__)

try:
    _K_LEVEL_SEQUENCE = BLOOM_LEVELS.k_level_sequence()
except Exception:
    _K_LEVEL_SEQUENCE = ("K1", "K2", "K3")

_BLOOM_SEQUENCE = BLOOM_LEVELS.sequence() or ("K1", "K2", "K3", "K4", "K5", "K6")
_LOWEST_BLOOM_LEVEL = _BLOOM_SEQUENCE[0]
_BLOOM_SEQUENCE_TEXT = ", ".join(_BLOOM_SEQUENCE)
_BLOOM_RANGE_TEXT = (
    f"{_BLOOM_SEQUENCE[0]}–{_BLOOM_SEQUENCE[-1]}"
    if len(_BLOOM_SEQUENCE) > 1
    else _BLOOM_SEQUENCE[0]
)
_K_LEVEL_OPTIONS_TEXT = ", ".join(_K_LEVEL_SEQUENCE)
from engines.competency import SKILL_REGISTRY, SkillDefinition
from engines.domain_adapter import DomainAdaptiveOrchestrator
from engines.elo import EloEngine
from engines.graph_path_planner import KnowledgeProcessOrchestrator, LearnerProfile, LearningGoal
from engines.progression import ProgressionEngine, ensure_progress_record
from engines.intervention_system import LearningInterventionSystem, LearningPattern
from knowledge_graph import ContentResource
from learning_path import AdaptiveLearningPathManager, LearningPathRecommendation
from schemas import LearningPatternRequest, InterventionResponse
from schemas import AssessmentResult, ChatResponse, LearnerModel, RubricCriterion, parse_json_safe

@asynccontextmanager
async def _lifespan(_: FastAPI):
    try:
        # Validate environment variables first
        from env_validation import validate_environment
        validate_environment()
        
        db.init()
        tutor.ensure_seed_items()
        logger.info("OpenAI-style params in use: %s | SEND_MAX_TOKENS: %s", 
                   _base_params(), SEND_MAX_TOKENS)
        _ensure_rag_ready()
        yield
    except Exception as e:
        logger.error("Failed to initialize application: %s", str(e), exc_info=True)
        raise


app = FastAPI(title="AILearnBuddy v4.4.6", version="4.4.6", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

TOKENS = {}

_PROTECTED_PATHS = frozenset({"/chat", "/privacy/export", "/privacy/delete"})


def _normalize_path(path: str) -> str:
    if not path or path == "/":
        return "/"
    return path.rstrip("/")


def _extract_token(header_value: Optional[str]) -> Optional[str]:
    if not header_value:
        return None
    candidate = header_value.strip()
    if not candidate:
        return None
    if " " in candidate:
        prefix, token = candidate.split(" ", 1)
        if prefix.lower() in {"bearer", "token"}:
            candidate = token.strip()
        else:
            candidate = token.strip() or prefix.strip()
    return candidate or None


def _authenticate_request(request: Request) -> Optional[str]:
    header_token = _extract_token(request.headers.get("authorization"))
    if header_token and header_token in TOKENS:
        return TOKENS[header_token]
    alt_header = request.headers.get("x-token")
    if alt_header and alt_header in TOKENS:
        return TOKENS[alt_header]
    query_token = request.query_params.get("token")
    if query_token and query_token in TOKENS:
        return TOKENS[query_token]
    return None


@app.middleware("http")
async def _enforce_token(request: Request, call_next):
    normalized_path = _normalize_path(request.url.path)
    if normalized_path in _PROTECTED_PATHS:
        user_id = _authenticate_request(request)
        if not user_id:
            return Response(
                status_code=401,
                content=json.dumps({"detail": "missing or invalid token"}),
                media_type="application/json",
            )
        request.state.user_id = user_id
    return await call_next(request)
_PBKDF2_ITERATIONS = 150_000
_PBKDF2_DIGEST = "sha256"

_INTERVENTION_SYSTEM = LearningInterventionSystem()
_PROGRESSION_ENGINE = ProgressionEngine(
    window_size=5,
    min_attempts=3,
    intervention_system=_INTERVENTION_SYSTEM,
)
LEARNING_PATH_MANAGER = AdaptiveLearningPathManager()
DOMAIN_ORCHESTRATOR = DomainAdaptiveOrchestrator()
KNOWLEDGE_ORCHESTRATOR = KnowledgeProcessOrchestrator(graph=DOMAIN_ORCHESTRATOR.graph)
KNOWLEDGE_GRAPH = DOMAIN_ORCHESTRATOR.graph
journey_tracker = journey.LearningJourneyTracker(graph=KNOWLEDGE_GRAPH)
ELO_ENGINE = EloEngine()

_LLM_LOGGER = logging.getLogger("ailb.llm")
if not _LLM_LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _LLM_LOGGER.addHandler(_handler)
_LLM_LOGGER.setLevel(logging.INFO)
_LLM_LOGGER.propagate = False

_DOMAIN_ALIASES = {
    "bpmn": "business_process",
    "business": "business_process",
    "business_process": "business_process",
    "process_management": "business_process",
    "process": "business_process",
    "math": "mathematics",
    "mathematics": "mathematics",
    "mathe": "mathematics",
    "language": "language_de_en",
    "german": "language_de_en",
    "de_en": "language_de_en",
    "language_de_en": "language_de_en",
    "language_zh_en": "language_zh_en",
    "zh_en": "language_zh_en",
    "mandarin": "language_zh_en",
    "chinese": "language_zh_en",
}

_BLOOM_INDEX = {level: idx for idx, level in enumerate(_BLOOM_SEQUENCE)}


_RAG_CORPUS_PATH = Path(os.getenv("RAG_CORPUS_PATH", str(Path("docs") / "rag_corpus.json")))
_GPT4ALL_RAG_URL = os.getenv("GPT4ALL_RAG_URL")
_RAG_SYSTEM_PROMPT = os.getenv(
    "RAG_SYSTEM_PROMPT",
    "You are an offline retrieval assistant. Answer the learner using the supplied knowledge snippets.",
)
_RAG_INITIALIZED = False
_RAG_DOCUMENTS: List[rag.Document] = []
_RAG_REMOTE_DOCUMENTS: List[rag.Document] = []
_RAG_STORE: Optional[rag.VectorStore] = None
_RAG_CHAIN: Optional[rag.ConversationalRetrievalChain] = None
_RAG_LOGGER = logging.getLogger("ailb.rag")


def _clean_bloom_level(level: Optional[str]) -> Optional[str]:
    if not level:
        return None
    key = str(level).strip().upper()
    return key if key in _BLOOM_INDEX else None


CHAT_PROMPT_VERSION = tutor.PROMPT_VERSION
CHAT_PROMPT_VARIANT = tutor.PROMPT_VARIANT
SIMPLE_CHAT_PROMPT_VERSION = "chat_simple.v1"
SIMPLE_CHAT_PROMPT_VARIANT = tutor.PROMPT_VARIANT

# ---------- Helpers ----------
def _generate_salt() -> str:
    return secrets.token_bytes(16).hex()


def _pbkdf2_hash(password: str, salt_hex: str) -> str:
    try:
        salt_bytes = bytes.fromhex(salt_hex)
    except ValueError:
        raise ValueError("Invalid salt for password hashing") from None
    return hashlib.pbkdf2_hmac(
        _PBKDF2_DIGEST,
        password.encode("utf-8"),
        salt_bytes,
        _PBKDF2_ITERATIONS,
    ).hex()


def _hash_password(password: str) -> tuple[str, str]:
    salt_hex = _generate_salt()
    return _pbkdf2_hash(password, salt_hex), salt_hex


def _verify_password(
    password: str,
    stored_hash: str,
    stored_salt: Optional[str],
) -> tuple[bool, Optional[tuple[str, str]]]:
    if stored_salt:
        try:
            derived = _pbkdf2_hash(password, stored_salt)
        except ValueError:
            return False, None
        return hmac.compare_digest(stored_hash or "", derived), None

    legacy_salt = os.getenv("AUTH_SALT", "local_salt")
    legacy_hash = hmac.new(
        legacy_salt.encode("utf-8"),
        password.encode("utf-8"),
        digestmod="sha256",
    ).hexdigest()
    if hmac.compare_digest(stored_hash or "", legacy_hash):
        new_hash, new_salt = _hash_password(password)
        return True, (new_hash, new_salt)
    return False, None

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


_ALLOWED_DIAGNOSES = {"conceptual", "procedural", "careless", "none"}


def _clamp_unit(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return max(0.0, min(1.0, number))


def _normalize_diagnosis(value: Any) -> Optional[str]:
    if value is None:
        return None
    candidate = str(value).strip().lower()
    if candidate not in _ALLOWED_DIAGNOSES:
        return None
    return None if candidate == "none" else candidate


def _normalize_history_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    normalized: list[dict[str, Any]] = []
    for entry in items:
        if entry is None:
            continue
        if isinstance(entry, dict):
            cleaned = {
                key: entry[key]
                for key in (
                    "bloom_level",
                    "delta",
                    "confidence",
                    "response_time_seconds",
                    "correct",
                    "score",
                    "note",
                    "summary",
                )
                if key in entry
            }
            if not cleaned:
                cleaned = dict(entry)
            normalized.append(cleaned)
        else:
            normalized.append({"note": str(entry)})
    return normalized


def _apply_history_update(user_id: str, topic: str, update: Any) -> None:
    if update in (None, "", False):
        return

    mode = "append"
    entries: list[dict[str, Any]] = []

    if isinstance(update, dict):
        mode = str(update.get("mode") or update.get("action") or "append").lower()
        raw_entries = update.get("entries") or update.get("entry")
        if raw_entries is not None:
            entries = _normalize_history_entries(raw_entries)
        else:
            payload = {
                key: value
                for key, value in update.items()
                if key not in {"mode", "action", "entries", "entry"}
            }
            if payload:
                entries = _normalize_history_entries(payload)
    elif isinstance(update, str):
        lowered = update.strip().lower()
        if lowered in {"clear", "reset"}:
            mode = "clear"
        else:
            entries = _normalize_history_entries(update)
    else:
        entries = _normalize_history_entries(update)

    try:
        state = LEARNING_PATH_MANAGER.get_state(user_id, topic)
    except Exception:
        state = {}

    history = state.get("history")
    if not isinstance(history, list):
        history = []

    if mode == "clear":
        history = []
    elif mode == "replace":
        history = entries
    else:
        history.extend(entries)

    state["history"] = history[-50:]

    try:
        db.upsert_learning_path_state(user_id, topic, state)
    except Exception:
        pass


def _handle_progression_action(user_id: str, topic: str, action_payload: Any) -> None:
    if not action_payload:
        return

    if isinstance(action_payload, dict):
        action_value = (
            action_payload.get("value")
            or action_payload.get("type")
            or action_payload.get("action")
            or action_payload.get("name")
        )
    else:
        action_value = action_payload

    value = str(action_value or "").strip().lower()
    if not value or value in {"none", "noop", "skip"}:
        return

    normalized = value.replace("_", ":")
    if normalized.startswith("progression"):
        try:
            ensure_progress_record(user_id, topic)
        except Exception:
            pass
        try:
            _PROGRESSION_ENGINE.evaluate(user_id, topic)
        except Exception:
            pass


def _coerce_assessment_result(
    payload: Any,
    *,
    user_id: str,
    topic: str,
    fallback_response: str,
    fallback_level: Optional[str],
    diagnosis: Optional[str] = None,
) -> AssessmentResult | None:
    data = payload
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    score = _clamp_unit(data.get("score"))
    if score is None:
        score = 0.0

    confidence_value = _clamp_unit(data.get("confidence"))
    if confidence_value is None:
        confidence_value = score

    bloom_candidate = _clean_bloom_level(data.get("bloom_level"))
    bloom_level = bloom_candidate or fallback_level or _LOWEST_BLOOM_LEVEL

    item_id = str(data.get("item_id") or _short_assessment_id("model"))
    response_text = str(data.get("response") or fallback_response or "")
    source_value = str(data.get("source") or "direct")
    prompt_version = str(data.get("prompt_version") or CHAT_PROMPT_VERSION)
    model_version = str(data.get("model_version") or tutor.MODEL_ID)

    latency_ms = data.get("latency_ms")
    tokens_in = data.get("tokens_in")
    tokens_out = data.get("tokens_out")

    diag_value = _normalize_diagnosis(data.get("diagnosis")) or diagnosis

    rubric_payload = data.get("rubric_criteria")
    rubric_list: list[RubricCriterion] = []
    if isinstance(rubric_payload, list):
        for idx, entry in enumerate(rubric_payload):
            try:
                if isinstance(entry, dict):
                    rubric_list.append(RubricCriterion.model_validate(entry))
                else:
                    rubric_list.append(
                        RubricCriterion(id=f"criterion_{idx}", score=float(entry))
                    )
            except Exception:
                continue

    self_assessment_raw = data.get("self_assessment")
    if isinstance(self_assessment_raw, str):
        self_assessment = self_assessment_raw.strip() or None
    else:
        self_assessment = None

    return AssessmentResult(
        user_id=user_id,
        domain=topic,
        item_id=item_id,
        bloom_level=bloom_level or _LOWEST_BLOOM_LEVEL,
        response=response_text,
        self_assessment=self_assessment,
        score=float(score),
        rubric_criteria=rubric_list,
        model_version=model_version,
        prompt_version=prompt_version,
        latency_ms=int(latency_ms) if isinstance(latency_ms, (int, float)) else None,
        tokens_in=int(tokens_in) if isinstance(tokens_in, (int, float)) else None,
        tokens_out=int(tokens_out) if isinstance(tokens_out, (int, float)) else None,
        confidence=float(confidence_value),
        diagnosis=diag_value,
        source=source_value if source_value in {"direct", "self_check", "heuristic"} else "direct",
    )


def _looks_non_english(text: str) -> bool:
    """Heuristic: detect when a response is predominantly non-English."""

    if not text:
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False

    ascii_letters = sum(1 for ch in letters if ch.isascii())
    non_ascii_letters = len(letters) - ascii_letters

    if non_ascii_letters == 0:
        return False

    if ascii_letters == 0:
        return True

    return non_ascii_letters > ascii_letters

def _extract_trailing_json(txt: str) -> tuple[str, dict]:
    m = re.search(r"(\{[\s\S]*\})\s*$", txt or "")
    if not m:
        raise ValueError("No JSON block found.")
    payload = json.loads(m.group(1))
    leading = (txt or "")[: m.start(1)].rstrip()
    return leading, payload


def _parse_chat_response_payload(text: str) -> tuple[str, ChatResponse]:
    """Return the narrative prefix and validated chat payload."""

    response = parse_json_safe(text, ChatResponse)
    leading = (text or "")
    match = re.search(r"(\{[\s\S]*\})\s*$", text or "")
    if match:
        leading = leading[: match.start(1)].rstrip()
    else:
        leading = leading.strip()
    return leading, response


def _attach_bloom_metadata(target: dict[str, Any], level: Optional[str]) -> None:
    info = BLOOM_LEVELS.get(level) if level else None
    if info is None:
        target.pop("bloom_level_label", None)
        target.pop("bloom_level_description", None)
        return
    target["bloom_level_label"] = info.label
    target["bloom_level_description"] = info.description


def _reference_sources(topic: str, bloom_level: Optional[str]) -> list[str]:
    try:
        refs = item_bank.collect_references(domain=topic, bloom_level=bloom_level, limit=5)
    except Exception:
        refs = []
    cleaned: list[str] = []
    for ref in refs:
        if ref.startswith("internal:"):
            cleaned.append(ref.replace("internal:", "Interne Ressource: "))
        else:
            cleaned.append(ref)
    return cleaned


def _ensure_rag_ready(force: bool = False) -> None:
    global _RAG_INITIALIZED, _RAG_DOCUMENTS, _RAG_REMOTE_DOCUMENTS, _RAG_STORE, _RAG_CHAIN
    if _RAG_INITIALIZED and not force:
        return

    documents: List[rag.Document] = []
    remote_documents: List[rag.Document] = []

    base_url = _rag_base_url()
    if base_url:
        try:
            remote_documents = rag.fetch_gpt4all_documents(base_url)
            documents.extend(remote_documents)
        except Exception as exc:  # pragma: no cover - defensive log for unexpected runtime errors
            _RAG_LOGGER.warning("RAG remote fetch failed: %s", exc)

    if not documents and _RAG_CORPUS_PATH.exists():
        try:
            documents.extend(rag.load_learning_materials(str(_RAG_CORPUS_PATH)))
        except Exception as exc:
            _RAG_LOGGER.warning("Failed to load RAG corpus %s: %s", _RAG_CORPUS_PATH, exc)

    if not documents and _RAG_CORPUS_PATH.is_dir():
        try:
            documents.extend(rag.load_learning_materials(str(_RAG_CORPUS_PATH), patterns=(".md", ".txt", ".json")))
        except Exception as exc:
            _RAG_LOGGER.warning("Failed to scan RAG directory %s: %s", _RAG_CORPUS_PATH, exc)

    if not documents:
        _RAG_DOCUMENTS = []
        _RAG_REMOTE_DOCUMENTS = []
        _RAG_STORE = None
        _RAG_CHAIN = None
        _RAG_INITIALIZED = True
        return

    chunk_size = _safe_int("RAG_CHUNK_SIZE", 420)
    chunk_overlap = _safe_int("RAG_CHUNK_OVERLAP", 80)
    try:
        chunks = rag.split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        store = rag.build_vector_store(chunks)
        chain = rag.ConversationalRetrievalChain(
            store,
            llm=_rag_llm,
            system_prompt=_RAG_SYSTEM_PROMPT,
        )
    except Exception as exc:
        _RAG_LOGGER.warning("Failed to initialise RAG store: %s", exc)
        _RAG_DOCUMENTS = []
        _RAG_REMOTE_DOCUMENTS = []
        _RAG_STORE = None
        _RAG_CHAIN = None
        _RAG_INITIALIZED = True
        return

    _RAG_DOCUMENTS = documents
    _RAG_REMOTE_DOCUMENTS = remote_documents
    _RAG_STORE = store
    _RAG_CHAIN = chain
    _RAG_INITIALIZED = True


def _canonical_domain(subject_id: Optional[str]) -> Optional[str]:
    if not subject_id:
        return None
    key = subject_id.lower().strip()
    if key in SKILL_REGISTRY:
        return key
    return _DOMAIN_ALIASES.get(key)


def _level_index(level: Optional[str]) -> int:
    if not level:
        return len(_BLOOM_SEQUENCE)
    return _BLOOM_INDEX.get(level, len(_BLOOM_SEQUENCE))


def _select_skill_definition(domain: str, bloom_level: str) -> Optional[SkillDefinition]:
    skills = SKILL_REGISTRY.get(domain)
    if not skills:
        return None
    for skill in skills:
        if skill.bloom_level == bloom_level:
            return skill
    target_idx = _level_index(bloom_level)
    sorted_candidates = sorted(skills, key=lambda skill: abs(_level_index(skill.bloom_level) - target_idx))
    return sorted_candidates[0] if sorted_candidates else None


def _normalise_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalise_metadata(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_normalise_metadata(v) for v in value)
    if isinstance(value, list):
        return [_normalise_metadata(v) for v in value]
    return value


def _mastered_nodes_for_domain(domain: str, levels: dict[str, Any], current_level: Optional[str]) -> set[str]:
    mastered: set[str] = set()
    current_idx = _level_index(current_level)
    for node in KNOWLEDGE_GRAPH.find_nodes(domain=domain):
        level_score = float(levels.get(node.bloom_level, 0.0) or 0.0)
        if level_score >= LEARNING_PATH_MANAGER.promotion_threshold or _level_index(node.bloom_level) <= current_idx:
            mastered.add(node.identifier)
    return mastered


def _build_event_log(user_id: str, domain: str) -> list[dict[str, Any]]:
    entries = db.list_journey(user_id=user_id, limit=200)
    events: list[dict[str, Any]] = []
    for entry in entries:
        op = entry.get("op")
        if op not in {"session_event", "standalone_event", "learning_event_recorded"}:
            continue
        payload = entry.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        subject = payload.get("subject_id") or payload.get("topic")
        if subject:
            mapped = _canonical_domain(str(subject))
            if mapped and mapped != domain:
                continue
        timestamp = payload.get("recorded_at") or entry.get("created_at")
        if not timestamp:
            continue
        details = payload.get("details")
        if details is None:
            details = {}
        elif not isinstance(details, dict):
            details = {"value": details}
        activity = str(payload.get("event_type") or payload.get("activity") or op)
        metadata = journey.enrich_event_metadata(
            details,
            subject_id=payload.get("subject_id") or subject or domain,
            skill_id=payload.get("skill_id"),
            competency_id=payload.get("competency_id"),
            score=payload.get("score"),
            event_type=activity,
            outcome=payload.get("outcome") or payload.get("status"),
            status=payload.get("status"),
        )
        metadata = {"op": op, **metadata}
        score = payload.get("score")
        if score is not None:
            metadata.setdefault("score", float(score))
        events.append(
            {
                "case_id": str(payload.get("session_id") or payload.get("event_id") or f"{user_id}:{entry.get('id')}") ,
                "activity": activity,
                "timestamp": str(timestamp),
                "skill_area": metadata.get("skill_id")
                or metadata.get("competency_id")
                or payload.get("subject_id")
                or domain,
                "metadata": metadata,
            }
        )
    return events


def _plan_learning_path(
    user_id: str,
    subject_id: str,
    domain: str,
    *,
    mastered_nodes: set[str],
    target_skill_id: str,
    target_bloom: str,
    preferences: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    profile = LearnerProfile(
        user_id=user_id,
        mastered_nodes=set(mastered_nodes),
        preferences=preferences or {},
    )
    goal = LearningGoal(domain=domain, skill_ids=[target_skill_id], target_bloom=target_bloom, metadata={"subject_id": subject_id})
    event_log = _build_event_log(user_id, domain)
    plan = KNOWLEDGE_ORCHESTRATOR.recommend_path(profile, [goal], event_log, end_activity="mastery")
    ordered_nodes = [
        {
            "id": node.identifier,
            "label": node.label,
            "bloom_level": node.bloom_level,
            "domain": node.domain,
            "metadata": _normalise_metadata(node.metadata),
        }
        for node in plan.ordered_nodes
    ]
    resource_map: dict[str, list[dict[str, Any]]] = {}
    for node_id, resources in plan.resources.items():
        serialised: list[dict[str, Any]] = []
        for resource in resources:
            if isinstance(resource, ContentResource):
                serialised.append(resource.to_dict())
            elif hasattr(resource, "resource_id"):
                serialised.append(
                    {
                        "resource_id": getattr(resource, "resource_id"),
                        "title": getattr(resource, "title", ""),
                        "uri": getattr(resource, "uri", ""),
                        "modality": getattr(resource, "modality", "unknown"),
                        "metadata": _normalise_metadata(getattr(resource, "metadata", {})),
                    }
                )
        if serialised:
            resource_map[node_id] = serialised
    return {
        "ordered_nodes": ordered_nodes,
        "resources": resource_map,
        "diagnostics": _normalise_metadata(plan.diagnostics),
        "insights": plan.insights,
        "skill_success": {skill: float(rate) for skill, rate in plan.skill_success.items()},
        "events_considered": len(event_log),
        "preference_matches": {
            node_id: matches
            for node_id, matches in plan.preference_matches.items()
            if matches
        },
        "preference_highlights": plan.preference_highlights,
        "preferences_applied": preferences or {},
    }


def _log_competency_recommendation(
    assessment: AssessmentResult,
    *,
    bloom_update: Optional[LearningPathRecommendation],
) -> None:
    subject_id = (assessment.domain or "").strip()
    canonical = _canonical_domain(subject_id)
    if not canonical:
        return

    try:
        state = LEARNING_PATH_MANAGER.get_state(assessment.user_id, subject_id)
    except Exception:
        state = {"levels": {}}
    levels = state.get("levels", {}) or {}
    current_level = (bloom_update.recommended_level if bloom_update else None) or state.get("current_level") or assessment.bloom_level or _LOWEST_BLOOM_LEVEL
    target_skill = _select_skill_definition(canonical, current_level)
    if not target_skill:
        return

    mastery_value = float(levels.get(target_skill.bloom_level, 0.0) or 0.0)
    recommendation = DOMAIN_ORCHESTRATOR.recommend(target_skill, mastery=mastery_value)

    mastered_nodes = _mastered_nodes_for_domain(canonical, levels, current_level)
    preferences = state.get("preferences") if isinstance(state.get("preferences"), dict) else {}
    plan = _plan_learning_path(
        assessment.user_id,
        subject_id,
        canonical,
        mastered_nodes=mastered_nodes,
        target_skill_id=target_skill.skill_id,
        target_bloom=target_skill.bloom_level,
        preferences=preferences,
    )

    metadata = _normalise_metadata(recommendation.metadata)
    if "mastery" not in metadata:
        metadata["mastery"] = round(mastery_value, 3)
    mastery_snapshot = {
        "levels": {lvl: float(val) for lvl, val in levels.items()},
        "current_level": current_level,
        "mastered_nodes": sorted(mastered_nodes),
    }
    reason = (
        f"Focus on {target_skill.label} at {target_skill.bloom_level} with a {recommendation.modality} activity"
        f" (mastery {metadata.get('mastery', mastery_value):.2f})."
    )
    payload: dict[str, Any] = {
        "subject_id": subject_id,
        "domain": canonical,
        "skill": target_skill.skill_id,
        "target_skill": target_skill.skill_id,
        "skill_label": target_skill.label,
        "modality": recommendation.modality,
        "activity": recommendation.modality,
        "reason": reason,
        "activity_prompt": recommendation.prompt,
        "feedback_prompt": recommendation.feedback_prompt,
        "metadata": metadata,
        "mastery_snapshot": mastery_snapshot,
        "plan": plan,
    }
    if bloom_update:
        payload["bloom_decision"] = {
            "current_level": bloom_update.current_level,
            "recommended_level": bloom_update.recommended_level,
            "action": bloom_update.action,
            "confidence": float(bloom_update.confidence),
            "reason": bloom_update.reason,
            "reason_code": bloom_update.reason_code,
            "evidence": bloom_update.evidence,
            "progress_by_level": bloom_update.progress_by_level,
        }
    db.log_journey_update(assessment.user_id, "suggest_next_item", payload)


def _build_explainable_response(
    user_id: str,
    subject_id: str,
    *,
    bloom_level: Optional[str],
    context_sources: Optional[list[str]] = None,
    learning_update: Optional[LearningPathRecommendation] = None,
) -> tuple[str, float, list[str], dict[str, Any]]:
    try:
        state = LEARNING_PATH_MANAGER.get_state(user_id, subject_id)
    except Exception:
        state = {}
    levels = state.get("levels", {}) or {}
    current_level = bloom_level or state.get("current_level") or _LOWEST_BLOOM_LEVEL
    base_conf = levels.get(current_level, 0.0)
    if learning_update:
        confidence = learning_update.confidence
        recommended = learning_update.recommended_level
        action = learning_update.action
        reason = learning_update.reason
        reason_code = learning_update.reason_code
        evidence = learning_update.evidence
    else:
        confidence = max(0.2, float(base_conf) if base_conf else 0.5)
        recommended = state.get("current_level", current_level)
        action = "stabilise" if recommended == current_level else "review"
        reason = state.get("last_reason") or "Individuelle Lernhistorie berücksichtigt."
        reason_code = state.get("last_reason_code", "stabilise_monitor")
        evidence = state.get("last_evidence")
    sources = list(context_sources or [])
    history = state.get("history") or []
    if history:
        sources.append(f"Lernhistorie ({len(history)} Einträge)")
    if not sources:
        sources = [f"Interne Lernhistorie {subject_id or 'unbekannt'}"]
    explanation = (
        f"Diese Antwort basiert auf {', '.join(sources)} und Ihren bisherigen Antworten "
        f"auf Niveau {current_level}. (Confidence: {confidence:.2f})"
    )
    learning_info = {
        "current_level": current_level,
        "recommended_level": recommended,
        "action": action,
        "reason": reason,
        "progress_by_level": {lvl: round(val, 4) for lvl, val in levels.items()},
    }
    if reason_code:
        learning_info["reason_code"] = reason_code
    if evidence:
        learning_info["evidence"] = evidence
    return explanation, round(confidence, 3), sources, learning_info

def _ends_clean(txt: str) -> bool:
    """
    Heuristic: an answer is "finished" if it ends with standard punctuation
    OR if a closed JSON/code block is present at the end.
    """
    if not txt:
        return False
    s = txt.rstrip()

    # Typical sentence endings
    if s.endswith((".", "?", "!", "\"", "”", "…", ")", "›", "»")):
        return True

    # Fully closed code fences?
    if "```" in s and s.count("```") % 2 == 0:
        return True

    # JSON/array heuristically closed?
    if s.endswith("}") or s.endswith("]"):
        open_curly = s.count("{")
        close_curly = s.count("}")
        open_brack = s.count("[")
        close_brack = s.count("]")
        if open_curly == close_curly and open_brack == close_brack:
            return True

    return False

def _safe_float(env_name: str, default: float) -> float:
    raw = os.getenv(env_name, "")
    try:
        return float(raw) if raw else default
    except Exception:
        return default

def _safe_int(env_name: str, default: int) -> int:
    raw = os.getenv(env_name, "")
    try:
        return int(raw) if raw else default
    except Exception:
        return default

SEND_MAX_TOKENS = os.getenv("SEND_MAX_TOKENS", "true").lower() == "true"


def _rag_max_tokens() -> int:
    return _safe_int("RAG_MAX_TOKENS", 256)


def _rag_base_url() -> Optional[str]:
    if _GPT4ALL_RAG_URL:
        return _GPT4ALL_RAG_URL
    base = tutor.GPT4ALL_URL.rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    if "/v1/" in base:
        return base.split("/v1/", 1)[0]
    return base


def truncate_messages(messages: list[dict], hard_limit_tokens: int = 2800) -> list[dict]:
    if not messages:
        return []

    limit = max(1, _safe_int("MAX_TOKENS_IN", hard_limit_tokens))
    approx_tokens = 0
    kept_reversed: list[dict] = []

    for msg in reversed(messages):
        content = str(msg.get("content") or "")
        token_estimate = max(1, (len(content) + 3) // 4) if content else 0
        if approx_tokens + token_estimate <= limit or not kept_reversed:
            kept_reversed.append(msg)
            approx_tokens += token_estimate
            continue

        remaining = limit - approx_tokens
        if not kept_reversed and remaining > 0 and content:
            allowed_chars = max(4, remaining * 4)
            truncated = content[-allowed_chars:]
            new_msg = dict(msg)
            new_msg["content"] = truncated
            kept_reversed.append(new_msg)
            approx_tokens = limit
        break

    if not kept_reversed:
        return [messages[-1]]
    return list(reversed(kept_reversed))


def _short_assessment_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


def self_check_regrade(
    user_id: str,
    topic: str,
    user_text: str,
    answer_text: str,
) -> AssessmentResult | None:
    """Run a lightweight self-check to reduce hallucinated assessment JSON."""

    prompt = (
        "You are a strict grader. Grade the assistant's answer to the user's prompt using this rubric:\n"
        "correctness(0/1), completeness(0/1), clarity(0/1). score=(c1+c2+c3)/3.\n"
        "Return ONLY valid JSON for AssessmentResult with: user_id, domain, item_id ('selfcheck-' + id),"
        f" bloom_level (one of {_K_LEVEL_OPTIONS_TEXT}), score, rubric_criteria, model_version, prompt_version, confidence,"
        " source='self_check'.\n"
        f"User prompt:\n{user_text}\n—\nAssistant answer:\n{answer_text}"
    )
    messages = [
        {"role": "system", "content": tutor.SYSTEM_TUTOR_JSON},
        {"role": "user", "content": prompt},
    ]
    # Evidence principle: a self-check pass lets the tutor re-evaluate its own answer,
    # which empirically cuts hallucinated grades before we trust the signal.
    try:
        content = _llm_call(
            messages,
            max_tokens=600,
            user_id=user_id,
            prompt_version="self_check.v1",
            prompt_variant="self_check",
            bloom_before=_LOWEST_BLOOM_LEVEL,
            path_decisions=["self_check", topic],
        )
    except Exception:
        return None
    try:
        result = parse_json_safe(content, AssessmentResult)
    except Exception:
        return None
    try:
        db.mark_last_llm_metric_validated(user_id)
    except Exception:
        pass
    result = result.model_copy(update={
        "user_id": user_id,
        "domain": topic,
        "item_id": result.item_id or _short_assessment_id("selfcheck"),
        "source": "self_check",
        "confidence": max(0.7, float(result.confidence or 0.0)),
    })
    if not result.model_version:
        result.model_version = tutor.MODEL_ID
    if not result.prompt_version:
        result.prompt_version = "self_check.v1"
    return result


def _microcheck_context_key(recent_answer: str | None) -> str | None:
    if not recent_answer:
        return None
    collapsed = " ".join(str(recent_answer).split())
    if not collapsed:
        return None
    return collapsed[:200]


def _microcheck_learning_focus(
    user_id: str | None,
    subject_id: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    if not user_id or not subject_id:
        return None, None

    try:
        state = LEARNING_PATH_MANAGER.get_state(user_id, subject_id) or {}
    except Exception:
        state = {}

    focus_label: str | None = None
    snapshot: dict[str, Any] = {}

    if isinstance(state, Mapping):
        progress = state.get("levels")
        if isinstance(progress, Mapping):
            snapshot["progress_by_level"] = {
                str(level): round(float(value), 4)
                for level, value in progress.items()
                if level
            }

        current_level = _clean_bloom_level(state.get("current_level"))
        if current_level:
            snapshot["current_level"] = current_level

        reason = str(state.get("last_reason") or "").strip()
        if reason:
            snapshot["reason"] = reason

        reason_code = str(state.get("last_reason_code") or "").strip()
        if reason_code:
            snapshot["reason_code"] = reason_code

        evidence = state.get("last_evidence")
        if evidence is not None:
            try:
                snapshot["evidence"] = json.loads(json.dumps(evidence, default=str))
            except Exception:
                snapshot["evidence"] = str(evidence)

        if reason and current_level:
            focus_label = f"{reason} ({current_level})"
        elif reason:
            focus_label = reason
        elif current_level:
            focus_label = f"Bloom-Level {current_level}"

    try:
        recommendations = db.list_recent_recommendations(user_id, limit=1)
    except Exception:
        recommendations = []

    if recommendations:
        entry = recommendations[0] or {}
        skill = entry.get("skill")
        reason = entry.get("reason")
        if skill:
            snapshot.setdefault("recommended_skill", skill)
        if reason:
            snapshot.setdefault("learning_recommendation", reason)
        if reason and skill:
            focus_label = f"{reason} ({skill})"
        elif reason and not focus_label:
            focus_label = reason
        elif skill and not focus_label:
            focus_label = skill

    if not snapshot:
        return focus_label, None
    return focus_label, snapshot


@lru_cache(maxsize=100)
def _cached_microcheck(
    topic: str,
    hint: str | None = None,
    bloom_level: str | None = None,
    context_key: str | None = None,
    long_term_focus: str | None = None,
    snapshot_key: str | None = None,
) -> dict[str, Any]:
    snapshot: dict[str, Any] | None = None
    if snapshot_key:
        try:
            snapshot = json.loads(snapshot_key)
        except Exception:
            snapshot = None
    return tutor.generate_microcheck(
        topic,
        hint,
        bloom_level=bloom_level,
        recent_answer=context_key,
        learning_focus=long_term_focus,
        learning_snapshot=snapshot,
    )


def cached_microcheck(
    topic: str,
    hint: str | None = None,
    *,
    bloom_level: str | None = None,
    recent_answer: str | None = None,
    user_id: str | None = None,
    subject_id: str | None = None,
) -> dict[str, Any]:
    """Return a cached microcheck result with defensive copying."""

    normalized_level = _clean_bloom_level(bloom_level)
    context_key = _microcheck_context_key(recent_answer)
    focus_label, snapshot = _microcheck_learning_focus(user_id, subject_id)
    snapshot_key = None
    if snapshot is not None:
        try:
            snapshot_key = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            snapshot_key = None
    return copy.deepcopy(
        _cached_microcheck(
            topic,
            hint,
            normalized_level,
            context_key,
            focus_label,
            snapshot_key,
        )
    )


async def async_self_check_regrade(
    user_id: str,
    topic: str,
    user_text: str,
    answer_text: str,
) -> AssessmentResult | None:
    return await asyncio.to_thread(
        self_check_regrade,
        user_id,
        topic,
        user_text,
        answer_text,
    )


async def async_cached_microcheck(
    topic: str,
    hint: str | None = None,
    *,
    bloom_level: str | None = None,
    recent_answer: str | None = None,
    user_id: str | None = None,
    subject_id: str | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        cached_microcheck,
        topic,
        hint,
        bloom_level=bloom_level,
        recent_answer=recent_answer,
        user_id=user_id,
        subject_id=subject_id,
    )


async def parallel_llm_assessment(
    user_id: str,
    topic: str,
    user_text: str,
    answer_text: str,
    *,
    need_microcheck: bool = True,
    bloom_level: str | None = None,
) -> dict[str, Any]:
    tasks: dict[str, asyncio.Task[Any]] = {
        "self_check": asyncio.create_task(
            async_self_check_regrade(user_id, topic, user_text, answer_text)
        )
    }
    if need_microcheck:
        tasks["microcheck"] = asyncio.create_task(
            async_cached_microcheck(
                topic,
                bloom_level=bloom_level,
                recent_answer=answer_text,
                user_id=user_id,
                subject_id=topic,
            )
        )

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    return {key: value for key, value in zip(tasks.keys(), results)}


def log_pending_assessment(
    user_id: str,
    topic: str,
    user_text: str,
    answer_text: str,
) -> None:
    payload = {
        "op_id": _short_assessment_id("pending"),
        "type": "assessment_pending",
        "inputs": {
            "user_text": user_text,
            "answer_text": answer_text,
            "topic": topic,
        },
        "decision": "needs_regrade",
        "rationale": "no structured JSON",
    }
    try:
        db.save_pending_op(user_id, topic, payload)
    except Exception:
        pass
    try:
        from xapi import emit as emit_xapi

        emit_xapi(
            user_id=user_id,
            verb="http://adlnet.gov/expapi/verbs/experienced",
            object_id=f"https://your.app/pending/{payload['op_id']}",
            context={"reason": "no_json", "topic": topic},
        )
    except Exception:
        pass

def on_assessment_saved(assessment_obj: AssessmentResult) -> Optional[LearningPathRecommendation]:
    """Record progression artefacts and update the adaptive path."""

    subject_id = (assessment_obj.domain or "").strip()
    if not subject_id:
        return None

    attempt_id = None
    try:
        attempt = db.record_quiz_attempt(
            user_id=assessment_obj.user_id,
            subject_id=subject_id,
            activity_id=assessment_obj.item_id,
            score=float(assessment_obj.score),
            max_score=1.0,
            pass_threshold=0.6,
            confidence=float(assessment_obj.confidence or 0.0),
            path=str(assessment_obj.source),
            diagnosis=assessment_obj.diagnosis,
            self_assessment=assessment_obj.self_assessment,
        )
        attempt_id = attempt.get("attempt_id") if isinstance(attempt, dict) else None
    except Exception:
        # The progression engine should not block chat responses.
        pass

    try:
        ensure_progress_record(assessment_obj.user_id, subject_id)
    except Exception:
        pass

    try:
        _PROGRESSION_ENGINE.evaluate(
            assessment_obj.user_id,
            subject_id,
            last_attempt_id=int(attempt_id) if attempt_id is not None else None,
        )
    except Exception:
        pass

    recommendation: Optional[LearningPathRecommendation] = None
    try:
        recommendation = LEARNING_PATH_MANAGER.update_from_assessment(assessment_obj)
    except Exception:
        pass
    try:
        _log_competency_recommendation(assessment_obj, bloom_update=recommendation)
    except Exception:
        pass
    return recommendation

def _base_params():
    """Only OpenAI-style fields that GPT4All understands."""
    return {
        "temperature": _safe_float("LLM_TEMPERATURE", 0.2),
        "top_p": _safe_float("LLM_TOP_P", 0.95),
    }

# ---------- Auto-Continue ----------
CONTINUE_PROMPT = os.getenv("CONTINUE_PROMPT", "Please continue and complete your answer.")
MAX_CONT_STEPS  = _safe_int("MAX_CONTINUE_STEPS", 2)
MIN_CHUNK_WORDS = _safe_int("MIN_CONT_WORDS", 60)

COPILOT_SYSTEM_PROMPT = f"""You are an instructional design copilot. Draft lesson plans that teachers can adapt.
Return JSON with keys: title, overview, sections (list of {{title, bloom_level, activity, duration}}),
checks_for_understanding (list), resources (list), and extensions (list of optional enrichment ideas).
Use Bloom level labels {_BLOOM_SEQUENCE_TEXT} where relevant."""

EVAL_PROBES = [
    {
        "id": "reasoning_1",
        "category": "reasoning",
        "prompt": "Compute 12 + 7 * 3 and explain your reasoning briefly.",
        "expected_keywords": ["33", "multiplication"],
        "max_tokens": 180,
    },
    {
        "id": "kb_qa_rag_1",
        "category": "knowledge_qa",
        "prompt": "In one sentence, what is the purpose of BPMN diagrams?",
        "expected_keywords": ["process", "model"],
        "max_tokens": 150,
    },
    {
        "id": "bpmn_path_1",
        "category": "bpmn",
        "prompt": "When would you prefer an XOR gateway over an AND gateway in BPMN?",
        "expected_keywords": ["exclusive", "one"],
        "max_tokens": 150,
    },
    {
        "id": "language_k1",
        "category": "language",
        "prompt": "Define 'metaphor' in one sentence and provide a short example.",
        "expected_keywords": ["comparison", "like"],
        "max_tokens": 180,
    },
]

def _llm_call(
    messages,
    max_tokens: Optional[int],
    *,
    user_id: Optional[str] = None,
    prompt_version: Optional[str] = None,
    prompt_variant: Optional[str] = None,
    request_id: Optional[str] = None,
    bloom_before: Optional[str] = None,
    bloom_after: Optional[str] = None,
    elo_delta: Optional[float] = None,
    path_decisions: Optional[Sequence[str]] = None,
):
    payload = {"model": tutor.MODEL_ID, "messages": messages, **_base_params()}
    if max_tokens is not None and SEND_MAX_TOKENS:
        payload["max_tokens"] = int(max_tokens)

    rp = os.getenv("LLM_REPEAT_PENALTY", "")
    if rp:
        try:
            payload["repeat_penalty"] = float(rp)
        except (TypeError, ValueError) as exc:
            logging.getLogger(__name__).error(
                "Invalid LLM_REPEAT_PENALTY value '%s': %s", rp, exc
            )
            raise HTTPException(
                status_code=500,
                detail="Invalid configuration for LLM repeat penalty.",
            ) from exc

    normalized_path = [str(step) for step in (path_decisions or ())]
    call_request_id = request_id or str(uuid4())
    start = time.perf_counter()
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    data: Any = None
    model_id = str(payload.get("model", tutor.MODEL_ID))
    active_variant = prompt_variant or tutor.PROMPT_VARIANT
    try:
        try:
            r = requests.post(
                tutor.GPT4ALL_URL,
                json=payload,
                timeout=_safe_int("LLM_TIMEOUT", 1800)
            )
            if r.status_code == 400:
                # Fallback: send minimal payload
                minimal = {"model": tutor.MODEL_ID, "messages": messages}
                if max_tokens is not None and SEND_MAX_TOKENS:
                    minimal["max_tokens"] = int(max_tokens)
                r2 = requests.post(tutor.GPT4ALL_URL, json=minimal, timeout=_safe_int("LLM_TIMEOUT", 1800))
                r2.raise_for_status()
                data = r2.json()
            else:
                r.raise_for_status()
                data = r.json()
        except requests.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"LLM-HTTP {e.response.status_code}: {e.response.text[:300]}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM error: {e}")

        usage = data.get("usage") if isinstance(data, dict) else None

        def _coerce_int(value: Any) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        if isinstance(usage, dict):
            tokens_in = _coerce_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
            tokens_out = _coerce_int(usage.get("completion_tokens") or usage.get("output_tokens"))

        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            try:
                return data["choices"][0]["text"]
            except Exception:
                raise HTTPException(status_code=502, detail=f"Unexpected LLM response: {data}")
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        try:
            db.record_llm_metric(
                user_id=user_id,
                model_id=model_id,
                prompt_version=prompt_version or "default",
                prompt_variant=active_variant,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                path_taken=" > ".join(normalized_path) if normalized_path else None,
            )
        except Exception:
            pass
        log_record = {
            "event": "llm_call",
            "request_id": call_request_id,
            "user_id": user_id,
            "prompt_version": prompt_version or "default",
            "prompt_variant": active_variant,
            "model": model_id,
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "bloom_before": bloom_before,
            "bloom_after": bloom_after,
            "elo_delta": elo_delta,
            "path_decisions": normalized_path,
        }
        try:
            _LLM_LOGGER.info(json.dumps(log_record, ensure_ascii=False))
        except Exception:
            _LLM_LOGGER.info(log_record)


def _rag_llm(prompt: str) -> str:
    messages = [
        {"role": "system", "content": _RAG_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return _llm_call(messages, _rag_max_tokens(), path_decisions=["rag"])
    except HTTPException:
        raise
    except Exception as exc:
        _RAG_LOGGER.warning("RAG LLM fallback engaged: %s", exc)
        return "Unable to reach the language model. Review the retrieved context for guidance."

def generate_with_continue(
    system,
    user_text,
    max_tokens: Optional[int],
    extra_context: str = "",
    *,
    user_id: Optional[str] = None,
    prompt_version: Optional[str] = None,
    prompt_variant: Optional[str] = None,
    allow_non_english: bool = False,
    preferred_language: str = "English",
    request_id: Optional[str] = None,
    bloom_before: Optional[str] = None,
    bloom_after: Optional[str] = None,
    elo_delta: Optional[float] = None,
    path_decisions: Optional[Sequence[str]] = None,
) -> str:
    request_id = request_id or str(uuid4())
    base_path = [str(step) for step in (path_decisions or ())]
    if not base_path:
        base_path = ["initial"]
    active_version = prompt_version or tutor.PROMPT_VERSION
    active_variant = prompt_variant or tutor.PROMPT_VARIANT
    language_directive = ""
    if not allow_non_english:
        language_directive = (
            f"\nRespond strictly in {preferred_language}. If the learner switches languages, still reply in {preferred_language} "
            "and paraphrase any foreign terms in English."
        )

    base_system = system + language_directive + (f"\n{extra_context}" if extra_context else "")
    messages = [
        {"role": "system", "content": base_system},
        {"role": "user",   "content": user_text}
    ]
    messages = truncate_messages(messages)
    content = _strip_think(
        _llm_call(
            messages,
            max_tokens,
            user_id=user_id,
            prompt_version=active_version,
            prompt_variant=active_variant,
            request_id=request_id,
            bloom_before=bloom_before,
            bloom_after=bloom_after,
            elo_delta=elo_delta,
            path_decisions=base_path,
        )
    )

    if not allow_non_english and _looks_non_english(content):
        enforced_system = base_system + (
            f"\nYou must provide the entire reply in {preferred_language}. Translate or restate the answer fully in {preferred_language}."
        )
        retry_messages = [
            {"role": "system", "content": enforced_system},
            {"role": "user", "content": user_text},
        ]
        retry_messages = truncate_messages(retry_messages)
        content = _strip_think(
            _llm_call(
                retry_messages,
                max_tokens,
                user_id=user_id,
                prompt_version=active_version,
                prompt_variant=active_variant,
                request_id=request_id,
                bloom_before=bloom_before,
                bloom_after=bloom_after,
                elo_delta=elo_delta,
                path_decisions=[*base_path, "language_retry"],
            )
        )
        if _looks_non_english(content):
            return (
                f"I'm sorry, but I can only respond in {preferred_language} right now. "
                f"Please ask again so I can assist you in {preferred_language}."
            )

    total = content

    # Early exit: already complete or JSON parseable → stop
    if _ends_clean(total):
        return total.strip()
    try:
        _extract_trailing_json(total)
        return total.strip()
    except Exception:
        pass

    for step in range(MAX_CONT_STEPS):
        if _ends_clean(total):
            break
        try:
            _extract_trailing_json(total)
            break
        except Exception:
            pass

        before_words = len(total.split())
        tail = total[-800:]
        cont_system = base_system + "\nRespond ONLY with the continuation, without repeating yourself."
        if not allow_non_english:
            cont_system += f" Ensure every sentence remains in {preferred_language}."
        messages = [
            {"role": "system", "content": cont_system},
            {"role": "assistant", "content": tail},
            {"role": "user",   "content": CONTINUE_PROMPT}
        ]
        messages = truncate_messages(messages)
        cont = _strip_think(
            _llm_call(
                messages,
                max_tokens,
                user_id=user_id,
                prompt_version=active_version,
                prompt_variant=active_variant,
                request_id=request_id,
                bloom_before=bloom_before,
                bloom_after=bloom_after,
                elo_delta=elo_delta,
                path_decisions=[*base_path, f"continuation_{step + 1}"],
            )
        )
        cont = cont.lstrip()
        if cont and not total.endswith("\n"):
            total += "\n"
        total += cont
        if len(total.split()) - before_words < MIN_CHUNK_WORDS:
            break
        if _ends_clean(total):
            break
    return total.strip()

# ---------- Schemas ----------
class RegisterBody(BaseModel):
    user_id: str
    email: Optional[str] = None
    password: str

class LoginBody(BaseModel):
    user_id: str
    password: str

class ChatBody(BaseModel):
    user_id: str
    topic: str = "general"
    text: str
    apply_mode: Literal["auto","review"] = "auto"
    max_tokens: Optional[int] = None

class SimpleChatBody(BaseModel):
    user_id: str
    text: str
    max_tokens: Optional[int] = None


class FeedbackBody(BaseModel):
    user_id: Optional[str] = None
    answer_id: str
    rating: Literal["up", "down", "flag"]
    comment: Optional[str] = None
    confidence: Optional[float] = None
    tags: Optional[list[str]] = None


class XAPIEmitBody(BaseModel):
    user_id: str
    verb: str
    object_id: str
    score: Optional[float] = None
    success: Optional[bool] = None
    response: Optional[Any] = None
    context: Optional[dict[str, Any]] = None


class PrivacyConsentBody(BaseModel):
    user_id: str
    consented: bool
    consent_text: Optional[str] = None

class BloomScoreBody(BaseModel):
    user_id: str
    topic: str
    rubric: dict


class CopilotPlanSection(BaseModel):
    title: str
    bloom_level: str
    activity: str
    duration: Optional[str] = None


class CopilotPlanDraft(BaseModel):
    title: str
    overview: str
    sections: list[CopilotPlanSection]
    checks_for_understanding: list[str] = []
    resources: list[str] = []
    extensions: list[str] = []


class CopilotPlanRequest(BaseModel):
    teacher_id: str
    topic: str
    objectives: list[str]
    constraints: Optional[str] = None
    teacher_notes: Optional[str] = None
    locale: Optional[str] = "en"


class CopilotModerationBody(BaseModel):
    plan_id: int
    moderator_id: str
    decision: Literal["approved", "needs_revision", "rejected"]
    rationale: Optional[str] = None
    flags: Optional[list[str]] = None
    status: Optional[str] = None


class EvalRequest(BaseModel):
    run_id: Optional[str] = None
    use_llm: bool = False
    sample_size: Optional[int] = None


class EvalAttemptBody(BaseModel):
    learner_id: str
    topic: str
    score: float
    max_score: float
    attempt_id: Optional[str] = None
    strategy: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    attempted_at: Optional[str] = None
    session_id: Optional[str] = None
    instrument_id: Optional[str] = None
    instrument_version: Optional[str] = None
    instrument: Optional[dict[str, Any]] = None


class DiagnoseStartBody(BaseModel):
    user_id: str
    skill: Optional[str] = None


class JourneySessionStartBody(BaseModel):
    user_id: str
    subject_id: Optional[str] = None
    session_type: str
    metadata: Optional[dict[str, Any]] = None


class JourneySessionEndBody(BaseModel):
    user_id: str
    session_id: str
    summary: Optional[dict[str, Any]] = None
    ended_at: Optional[str] = None


class JourneyDiagnosticStartBody(BaseModel):
    user_id: str
    subject_id: Optional[str] = None
    skill: Optional[str] = None
    limit: Optional[int] = 3


class JourneyEventBody(BaseModel):
    user_id: str
    subject_id: Optional[str] = None
    event_type: str
    lesson_id: Optional[str] = None
    score: Optional[float] = None
    details: Optional[dict[str, Any]] = None
    session_id: Optional[str] = None
    skill_id: Optional[str] = None
    competency_id: Optional[str] = None
    outcome: Optional[str] = None
    status: Optional[str] = None


class RagQueryBody(BaseModel):
    question: str
    k: int = Field(default=3, ge=1, le=10)
    chat_history: List[Tuple[str, str]] = Field(default_factory=list)


class LearningPathOverrideBody(BaseModel):
    user_id: str
    subject_id: str
    target_level: Optional[str] = None
    notes: Optional[str] = None
    expires_at: Optional[str] = None
    applied_by: Optional[str] = Field(default=None, max_length=128)
    metadata: dict[str, Any] | None = None
    window_days: int | None = Field(default=7, ge=1, le=90)

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/rag/documents")
def rag_documents():
    _ensure_rag_ready()
    documents = []
    for doc in _RAG_DOCUMENTS:
        metadata = dict(doc.metadata or {})
        title = metadata.get("title") or metadata.get("name")
        tags = metadata.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        preview = doc.content.strip().replace("\n", " ")
        preview_text = preview[:220] + ("…" if len(preview) > 220 else "")
        documents.append(
            {
                "source": doc.source,
                "title": title,
                "tags": tags,
                "metadata": metadata,
                "preview": preview_text,
            }
        )
    return {
        "count": len(documents),
        "remote_count": len(_RAG_REMOTE_DOCUMENTS),
        "documents": documents,
        "initialized": _RAG_INITIALIZED,
    }


@app.post("/rag/query")
def rag_query(body: RagQueryBody):
    _ensure_rag_ready()
    if _RAG_CHAIN is None or _RAG_STORE is None:
        raise HTTPException(status_code=503, detail="RAG corpus unavailable.")

    try:
        result = _RAG_CHAIN.invoke(body.question, chat_history=body.chat_history, k=body.k)
    except HTTPException:
        raise
    except Exception as exc:
        _RAG_LOGGER.error("RAG query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(exc)}")

    context_payload = [
        {
            "id": chunk.id,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "score": chunk.score,
        }
        for chunk in result.get("context", [])
    ]

    return {
        "answer": result.get("answer"),
        "context": context_payload,
        "used_llm": bool(result.get("answer")),
    }


# ---------- Auth ----------
@app.post("/auth/register")
def auth_register(body: RegisterBody):
    if db.get_user_auth(body.user_id):
        raise HTTPException(status_code=400, detail="user_id exists")
    email = (body.email or "").strip() or None
    if email and db.get_user_by_email(email):
        raise HTTPException(status_code=400, detail="email exists")
    pw_hash, pw_salt = _hash_password(body.password)
    db.create_user(body.user_id, email, pw_hash, pw_salt)
    return {"ok": True}

@app.post("/auth/login")
def auth_login(body: LoginBody):
    row = db.get_user_auth(body.user_id)
    if not row:
        raise HTTPException(status_code=401, detail="invalid credentials")

    valid, upgrade = _verify_password(body.password, row["pw_hash"], row["pw_salt"])
    if not valid:
        raise HTTPException(status_code=401, detail="invalid credentials")
    if upgrade:
        db.update_user_password(body.user_id, upgrade[0], upgrade[1])
    token = secrets.token_urlsafe(24)
    TOKENS[token] = body.user_id
    return {"token": token, "user_id": body.user_id}

# ---------- Tutor Chat ----------
@app.post("/chat")
async def chat(body: ChatBody):
    db.ensure_user(body.user_id)
    followup_state = {}
    learning_state = {}
    
    async with asyncio.Lock():
        try:
            followup_state = db.get_followup_state(body.user_id, body.topic) or {}
        except Exception:
            logger.warning("Failed to get followup state for user %s topic %s", body.user_id, body.topic)
            followup_state = {}
        try:
            learning_state = LEARNING_PATH_MANAGER.get_state(body.user_id, body.topic)
        except Exception:
            logger.warning("Failed to get learning state for user %s topic %s", body.user_id, body.topic)
            learning_state = {}
    tracked_level = _clean_bloom_level(learning_state.get("current_level"))
    prompt_level = tracked_level or _LOWEST_BLOOM_LEVEL
    raw_struggles = learning_state.get("struggles") or []
    if isinstance(raw_struggles, dict):
        struggle_iterable = raw_struggles.values()
    elif isinstance(raw_struggles, (list, tuple, set)):
        struggle_iterable = raw_struggles
    elif isinstance(raw_struggles, str):
        struggle_iterable = [raw_struggles]
    else:
        struggle_iterable = []
    struggles = []
    for entry in struggle_iterable:
        text = str(entry).strip()
        if text:
            struggles.append(text)
    microcheck_directive = ""
    prefetched_microcheck: dict[str, Any] | None = None
    accepted_assessment: AssessmentResult | None = None
    assessment_already_saved = False
    latest_learning_update: Optional[LearningPathRecommendation] = None
    reported_bloom_level: Optional[str] = tracked_level
    level_source: Optional[str] = "learning_state" if tracked_level else None

    if followup_state.get("needs_assessment"):
        question = followup_state.get("microcheck_question") or ""
        answer_key = followup_state.get("microcheck_answer_key") or ""
        raw_rubric = followup_state.get("microcheck_rubric")
        rubric = raw_rubric if raw_rubric is not None else []
        state = (followup_state.get("microcheck_source") or "pending").lower()

        if state == "awaiting_user":
            score = tutor.score_microcheck(body.text, answer_key, raw_rubric)
            accepted_assessment = AssessmentResult(
                user_id=body.user_id,
                domain=body.topic,
                item_id=_short_assessment_id("microcheck"),
                bloom_level=_LOWEST_BLOOM_LEVEL,
                response=body.text,
                score=float(score),
                rubric_criteria=[RubricCriterion(id="heuristic", score=float(score))],
                model_version=tutor.MODEL_ID,
                prompt_version="microcheck.heuristic.v1",
                confidence=0.5,
                source="heuristic",
            )
            db.save_assessment_result(accepted_assessment)
            update = on_assessment_saved(accepted_assessment)
            if update is not None:
                latest_learning_update = update
            sanitized_level = _clean_bloom_level(accepted_assessment.bloom_level)
            if sanitized_level:
                reported_bloom_level = sanitized_level
                level_source = "assessment"
            assessment_already_saved = True
            try:
                db.clear_followup_state(body.user_id, body.topic)
            except Exception:
                pass
            followup_state = {}
        else:
            if not question:
                generated = cached_microcheck(
                    body.topic,
                    bloom_level=prompt_level,
                    recent_answer=None,
                    user_id=body.user_id,
                    subject_id=body.topic,
                )
                question = generated.get("question", "Provide a one-sentence recap of the concept.")
                answer_key = generated.get("answer_key") or generated.get("key") or "definition"
                rubric = generated.get("rubric") or []
            microcheck_directive = f"Before answering, ask this 1-line check and grade it: {question}"
            if state != "awaiting_user":
                try:
                    db.set_needs_assessment(
                        body.user_id,
                        body.topic,
                        True,
                        microcheck={
                            "question": question,
                            "answer_key": answer_key,
                            "expected": answer_key,
                            "rubric": rubric,
                            "source": "awaiting_user",
                        },
                    )
                except Exception:
                    pass

    rows = db.get_prompts_for_topic(body.topic, limit=2)
    context_prompts = "\n".join([f"- {r['prompt_text']}" for r in rows])

    max_tokens = int(body.max_tokens) if body.max_tokens is not None else _safe_int("MAX_TOKENS", 600)
    topic_key = (body.topic or "bpmn").strip().lower()
    canonical_theme = _canonical_domain(body.topic)
    if canonical_theme and canonical_theme in tutor.SUPPORTED_THEMES:
        theme_key = canonical_theme
    elif topic_key in tutor.SUPPORTED_THEMES:
        theme_key = topic_key
    elif canonical_theme:
        theme_key = canonical_theme
    else:
        theme_key = topic_key or "bpmn"

    system_prompt = tutor.build_system_prompt(theme_key, prompt_level, struggles)

    system = (
        system_prompt
        + "\n"
        + tutor.SYSTEM_TUTOR_JSON
        + "\n"
        + tutor.JSON_RESPONSE_DIRECTIVE
        + f"\nContext prompts ({body.topic}):\n{context_prompts}"
    )
    allow_non_english = "language" in (body.topic or "").lower()
    path_notes = ["chat"]
    if followup_state.get("needs_assessment"):
        path_notes.append("followup_required")
    path_notes.append(f"bloom_{prompt_level}" if prompt_level else "bloom_unknown")
    if struggles:
        path_notes.append("struggles_present")
    content = generate_with_continue(
        system,
        body.text,
        max_tokens,
        user_id=body.user_id,
        prompt_version=CHAT_PROMPT_VERSION,
        prompt_variant=CHAT_PROMPT_VARIANT,
        extra_context=microcheck_directive,
        allow_non_english=allow_non_english,
        bloom_before=prompt_level,
        path_decisions=path_notes,
    )

    applied, pending = [], []
    raw_obj: dict[str, Any] | None = None
    raw_response_text = content
    answer = content.strip()
    cleaned_answer = ""
    text_without_json = content
    parsed_payload: ChatResponse | None = None
    parse_errors: list[Exception] = []

    # JSON parsing and retry behavior is documented in detail at docs/json/README.md.
    for attempt in range(2):
        try:
            text_without_json, parsed_payload = _parse_chat_response_payload(content)
            break
        except (ValidationError, ValueError, TypeError) as exc:
            parse_errors.append(exc)
            if attempt == 0:
                retry_context_parts: list[str] = []
                if microcheck_directive:
                    retry_context_parts.append(microcheck_directive.strip())
                retry_context_parts.append(
                    "Your previous reply failed JSON schema validation. Provide the teaching reply followed by a single JSON object that matches the required tutor schema. Respond with a single JSON object only, with no text before or after. The object must match the required schema."
                    f" Error: {exc}"
                )
                retry_extra_context = "\n".join(part for part in retry_context_parts if part)
                content = generate_with_continue(
                    system,
                    body.text,
                    max_tokens,
                    user_id=body.user_id,
                    prompt_version=CHAT_PROMPT_VERSION,
                    prompt_variant=CHAT_PROMPT_VARIANT,
                    extra_context=retry_extra_context,
                    allow_non_english=allow_non_english,
                    bloom_before=prompt_level,
                    path_decisions=[*path_notes, "schema_retry"],
                )
                raw_response_text = content
                continue
            break

    if parsed_payload is None:
        detail = "Tutor response was malformed and could not be parsed."
        if parse_errors:
            detail = f"{detail} ({parse_errors[-1]})"
        raise HTTPException(status_code=502, detail=detail)

    obj = parsed_payload.model_dump(mode="python")
    try:
        db.mark_last_llm_metric_validated(body.user_id)
    except Exception:
        pass
    raw_obj = obj

    try:
        ops = obj.get("db_ops", [])
        if not isinstance(ops, list):
            ops = []
        for op in ops:
            if not isinstance(op, dict):
                continue
            name = op.get("op") or op.get("operation")
            payload = op.get("payload", {}) or {}

            if name not in {"add_prompt", "log_rationale", "suggest_next_item"}:
                continue

            db.log_journey_update(body.user_id, name, payload)

            if body.apply_mode == "review":
                pending.append(op)
                continue

            if name == "add_prompt":
                topic = payload.get("topic")
                ptxt = (payload.get("prompt_text", "") or "").strip()

                bad_markers = ["español", "spanish", "français", "english", "inglés", "中文", "日本語", "русский", "arabic", "العربية"]
                if topic != body.topic:
                    pending.append({"op": name, "payload": payload, "reason": "topic_mismatch"})
                    continue
                if any(bm in ptxt.lower() for bm in bad_markers):
                    pending.append({"op": name, "payload": payload, "reason": "language_mismatch"})
                    continue
                if isinstance(topic, str) and isinstance(ptxt, str) and 0 < len(ptxt) <= 1000:
                    db.add_prompt(topic, ptxt, source="generated")
                    applied.append(op)
            else:
                applied.append(op)

        base_diagnosis = _normalize_diagnosis(obj.get("diagnosis"))
        if base_diagnosis is not None:
            obj["diagnosis_normalized"] = base_diagnosis

        action_payload = obj.get("action")
        if action_payload:
            _handle_progression_action(body.user_id, body.topic, action_payload)

        history_update_payload = obj.get("history_update")
        if history_update_payload:
            _apply_history_update(body.user_id, body.topic, history_update_payload)

        def _normalize_micro_text(value: Any) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        micro_question_text = _normalize_micro_text(obj.get("microcheck_question"))
        micro_expected_text = _normalize_micro_text(
            obj.get("microcheck_expected") or obj.get("microcheck_answer_key")
        )
        micro_given_text = _normalize_micro_text(obj.get("microcheck_given"))
        micro_score_value = _clamp_unit(obj.get("microcheck_score"))

        if any(
            v is not None
            for v in (micro_question_text, micro_expected_text, micro_given_text, micro_score_value)
        ):
            needs_followup = micro_score_value is None
            micro_payload: dict[str, Any] = {
                "question": micro_question_text,
                "answer_key": micro_expected_text,
                "expected": micro_expected_text,
                "given": micro_given_text,
                "score": micro_score_value,
                "source": "json",
            }
            rubric_payload = obj.get("microcheck_rubric")
            if rubric_payload is not None:
                micro_payload["rubric"] = rubric_payload
            try:
                db.set_needs_assessment(
                    body.user_id,
                    body.topic,
                    bool(needs_followup),
                    microcheck=micro_payload,
                )
            except Exception:
                pass
            followup_state = dict(followup_state or {})
            followup_state["needs_assessment"] = bool(needs_followup)
            if micro_question_text is not None:
                followup_state["microcheck_question"] = micro_question_text
            if micro_expected_text is not None:
                followup_state["microcheck_answer_key"] = micro_expected_text
                followup_state["microcheck_expected"] = micro_expected_text
            if micro_given_text is not None:
                followup_state["microcheck_given"] = micro_given_text
            followup_state["microcheck_score"] = micro_score_value
            followup_state["microcheck_source"] = "json"
            followup_state["microcheck_rubric"] = {
                "expected": micro_expected_text,
                "given": micro_given_text,
                "score": micro_score_value,
            }
            obj["microcheck_state"] = {
                "question": micro_question_text,
                "expected": micro_expected_text,
                "given": micro_given_text,
                "score": micro_score_value,
            }

        answer = obj.get("answer") or text_without_json.strip() or content.strip()
        cleaned_answer = _strip_think(answer)
        top_level_self_assessment = _normalize_micro_text(obj.get("self_assessment"))
        if accepted_assessment is None:
            fallback_level = reported_bloom_level or prompt_level or _LOWEST_BLOOM_LEVEL
            assessment_payload = obj.get("assessment_result") or obj.get("assessment")
            if assessment_payload:
                if (
                    isinstance(assessment_payload, dict)
                    and top_level_self_assessment
                    and not _normalize_micro_text(assessment_payload.get("self_assessment"))
                ):
                    assessment_payload = dict(assessment_payload)
                    assessment_payload["self_assessment"] = top_level_self_assessment
                accepted_assessment = _coerce_assessment_result(
                    assessment_payload,
                    user_id=body.user_id,
                    topic=body.topic,
                    fallback_response=cleaned_answer or answer or body.text,
                    fallback_level=fallback_level,
                    diagnosis=base_diagnosis,
                )
                if (
                    accepted_assessment
                    and top_level_self_assessment
                    and not accepted_assessment.self_assessment
                ):
                    accepted_assessment.self_assessment = top_level_self_assessment
        suggested_level = _clean_bloom_level(obj.get("bloom_level"))
        if suggested_level and level_source in {None, "fallback"}:
            reported_bloom_level = suggested_level
            level_source = "model_suggestion"

        if accepted_assessment is None and cleaned_answer:
                need_microcheck = not followup_state.get("needs_assessment")
                regraded: AssessmentResult | None = None
                if need_microcheck:
                    fallback_regrade = False
                    fallback_microcheck = False
                    try:
                        parallel_results = asyncio.run(
                        parallel_llm_assessment(
                            body.user_id,
                            body.topic,
                            body.text,
                            cleaned_answer,
                            need_microcheck=True,
                            bloom_level=prompt_level,
                        )
                        )
                    except RuntimeError:
                        parallel_results = {}
                        fallback_regrade = True
                        fallback_microcheck = True
                    except Exception:
                        parallel_results = {}
                        fallback_regrade = True
                        fallback_microcheck = True
                    else:
                        maybe_regraded = parallel_results.get("self_check")
                        if isinstance(maybe_regraded, Exception):
                            fallback_regrade = True
                        else:
                            regraded = maybe_regraded
                        maybe_microcheck = parallel_results.get("microcheck")
                        if isinstance(maybe_microcheck, Exception):
                            fallback_microcheck = True
                        else:
                            prefetched_microcheck = maybe_microcheck
                    if fallback_regrade:
                        regraded = self_check_regrade(
                            body.user_id, body.topic, body.text, cleaned_answer
                        )
                    if fallback_microcheck:
                        prefetched_microcheck = cached_microcheck(
                            body.topic,
                            bloom_level=prompt_level,
                            recent_answer=cleaned_answer,
                            user_id=body.user_id,
                            subject_id=body.topic,
                        )
                else:
                    regraded = self_check_regrade(
                        body.user_id, body.topic, body.text, cleaned_answer
                    )

                if regraded is not None:
                    accepted_assessment = regraded
                    if base_diagnosis and not accepted_assessment.diagnosis:
                        accepted_assessment.diagnosis = base_diagnosis
                    sanitized_level = _clean_bloom_level(regraded.bloom_level)
                    if sanitized_level:
                        reported_bloom_level = sanitized_level
                        level_source = "assessment"

        if accepted_assessment and not assessment_already_saved:
            if base_diagnosis and not accepted_assessment.diagnosis:
                accepted_assessment.diagnosis = base_diagnosis
            db.save_assessment_result(accepted_assessment)
            update = on_assessment_saved(accepted_assessment)
            if update is not None:
                latest_learning_update = update
            sanitized_level = _clean_bloom_level(accepted_assessment.bloom_level)
            if sanitized_level:
                reported_bloom_level = sanitized_level
                level_source = "assessment"
            assessment_already_saved = True
            raw_obj = dict(raw_obj or {})
            raw_obj["assessment_result"] = accepted_assessment.model_dump()
            try:
                db.clear_followup_state(body.user_id, body.topic)
            except Exception:
                pass
    except Exception:
        answer = content.strip()
        cleaned_answer = _strip_think(answer)

    if not cleaned_answer:
        cleaned_answer = _strip_think(answer)

    if not reported_bloom_level:
        reported_bloom_level = prompt_level or _LOWEST_BLOOM_LEVEL
        level_source = level_source or "fallback"

    context_sources = _reference_sources(body.topic, reported_bloom_level)
    if rows:
        context_sources.append(f"Promptdatenbank ({len(rows)} Lernimpulse)")

    explanation, confidence_score, used_sources, learning_info = _build_explainable_response(
        body.user_id,
        body.topic,
        bloom_level=reported_bloom_level,
        context_sources=context_sources,
        learning_update=latest_learning_update,
    )

    if accepted_assessment is None:
        existing_micro = bool(
            followup_state.get("microcheck_question")
            or followup_state.get("microcheck_answer_key")
            or followup_state.get("microcheck_expected")
        )
        if not followup_state.get("needs_assessment") and not existing_micro:
            generated = prefetched_microcheck or cached_microcheck(
                body.topic,
                bloom_level=reported_bloom_level,
                recent_answer=cleaned_answer,
                user_id=body.user_id,
                subject_id=body.topic,
            )
            try:
                db.set_needs_assessment(
                    body.user_id,
                    body.topic,
                    True,
                    microcheck={
                        "question": generated.get("question"),
                        "answer_key": generated.get("answer_key") or generated.get("key"),
                        "expected": generated.get("answer_key") or generated.get("key"),
                        "rubric": generated.get("rubric"),
                        "source": "pending",
                    },
                )
            except Exception:
                pass
        log_pending_assessment(body.user_id, body.topic, body.text, cleaned_answer)

    try:
        db.record_chat_ops(
            body.user_id,
            body.topic,
            body.text,
            cleaned_answer,
            raw_obj,
            applied,
            pending,
            raw_response=raw_response_text,
        )
    except Exception:
        pass

    return {
        "answer_text": cleaned_answer,
        "explanation": explanation,
        "confidence": confidence_score,
        "sources": used_sources,
        "learning_path": learning_info,
        "applied_ops": applied,
        "pending_ops": pending,
    }


@app.post("/feedback")
def receive_feedback(body: FeedbackBody):
    user_id = (body.user_id or "").strip() or None
    try:
        feedback_id = db.store_feedback(
            user_id,
            body.answer_id,
            body.rating,
            body.comment or "",
            confidence=body.confidence,
            tags=body.tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    summary = db.aggregate_feedback(answer_id=body.answer_id)
    return {"status": "success", "feedback_id": feedback_id, "summary": summary}


@app.post("/xapi/emit")
def emit_xapi_statement(body: XAPIEmitBody):
    try:
        xapi.emit(
            user_id=body.user_id,
            verb=body.verb,
            object_id=body.object_id,
            score=body.score,
            success=body.success,
            response=body.response,
            context=body.context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logging.getLogger("ailb.xapi.api").exception("Failed to emit xAPI statement: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to persist xAPI statement") from exc
    return {"status": "ok"}


# ---------- Simple Chat ----------
@app.post("/chat_simple")
def chat_simple(body: SimpleChatBody):
    db.ensure_user(body.user_id)
    max_tokens = int(body.max_tokens) if body.max_tokens is not None else _safe_int("SIMPLE_MAX_TOKENS", 800)
    system = "You are a tutor. Respond clearly and helpfully without JSON or <think>."
    content = generate_with_continue(
        system,
        body.text,
        max_tokens,
        user_id=body.user_id,
        prompt_version=SIMPLE_CHAT_PROMPT_VERSION,
        prompt_variant=SIMPLE_CHAT_PROMPT_VARIANT,
        path_decisions=["chat_simple"],
    )
    answer = _strip_think(content)
    explanation, confidence_score, used_sources, learning_info = _build_explainable_response(
        body.user_id,
        "general",
        bloom_level=None,
        context_sources=["Konversationsmodus"],
        learning_update=None,
    )
    return {
        "answer_text": answer,
        "explanation": explanation,
        "confidence": confidence_score,
        "sources": used_sources,
        "learning_path": learning_info,
    }

# ---------- Profile ----------
@app.get("/profile")
def profile(user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    db.ensure_user(user_id)

    start_batch = time.perf_counter()
    profile_batch = db.batch_learner_profile_data(user_id)
    batch_duration = time.perf_counter() - start_batch

    legacy_duration: Optional[float] = None
    compare_flag = os.getenv("PROFILE_DB_COMPARE", "").lower()
    if compare_flag in {"1", "true", "yes", "on"}:
        legacy_start = time.perf_counter()
        legacy_mastery = db.list_mastery(user_id=user_id, limit=200)
        legacy_bloom = db.list_bloom_progress(user_id=user_id, limit=200)
        legacy_topics = {row["skill"] for row in legacy_mastery if row["skill"]}
        legacy_topics.update(row["topic"] for row in legacy_bloom if row["topic"])
        legacy_followups = {topic: db.get_followup_state(user_id, topic) for topic in legacy_topics}
        legacy_assessments = db.list_recent_assessments(user_id, limit=50)
        legacy_duration = time.perf_counter() - legacy_start
        _ = (legacy_mastery, legacy_bloom, legacy_followups, legacy_assessments)

    logger.info(
        "profile() learner data timings: batch=%.4fs legacy=%s",
        batch_duration,
        f"{legacy_duration:.4f}s" if legacy_duration is not None else "skipped",
    )

    mastery_rows = profile_batch.get("mastery", [])
    skills: dict[str, dict[str, Any]] = {}
    for row in mastery_rows:
        skill_name = row["skill"]
        theta = float(row["level"]) if row["level"] is not None else None
        p_know = 1 / (1 + math.exp(-theta)) if theta is not None else None
        skills[skill_name] = {
            "theta": theta,
            "p_know": round(p_know, 3) if p_know is not None else None,
            "updated_at": row["updated_at"],
            "recent_attempts": [],
        }

    history_rows = db.list_bloom_progress_history(user_id=user_id, limit=500)
    history_by_topic: dict[str, list[dict[str, Any]]] = {}
    for row in history_rows:
        topic = row["topic"]
        history_entry = {
            "previous_level": row["previous_level"],
            "new_level": row["new_level"],
            "k_level": row["k_level"],
            "reason": row["reason"],
            "average_score": row["average_score"],
            "attempts_considered": row["attempts_considered"],
            "created_at": row["created_at"],
        }
        history_by_topic.setdefault(topic, []).append(history_entry)
    for events in history_by_topic.values():
        events.sort(key=lambda item: item.get("created_at") or "", reverse=True)

    bloom_progress_rows = profile_batch.get("bloom_progress", [])
    bloom_progress: dict[str, dict[str, Any]] = {}
    for row in bloom_progress_rows:
        topic = row["topic"]
        history = history_by_topic.pop(topic, None)
        bloom_entry = {
            "current_level": row["current_level"],
            "updated_at": row["updated_at"],
            "history": history or [],
        }
        _attach_bloom_metadata(bloom_entry, bloom_entry.get("current_level"))
        bloom_progress[topic] = bloom_entry
        entry = skills.setdefault(topic, {"theta": None, "p_know": None, "updated_at": row["updated_at"], "recent_attempts": []})
        entry["bloom_level"] = row["current_level"]
        entry["bloom_updated_at"] = row["updated_at"]
        _attach_bloom_metadata(entry, entry.get("bloom_level"))

    for topic, history in history_by_topic.items():
        if not history:
            continue
        latest = history[0]
        bloom_progress[topic] = {
            "current_level": latest.get("new_level"),
            "updated_at": latest.get("created_at"),
            "history": history,
        }
        _attach_bloom_metadata(bloom_progress[topic], bloom_progress[topic].get("current_level"))
        entry = skills.setdefault(topic, {"theta": None, "p_know": None, "updated_at": latest.get("created_at"), "recent_attempts": []})
        entry.setdefault("bloom_level", latest.get("new_level"))
        entry.setdefault("bloom_updated_at", latest.get("created_at"))
        _attach_bloom_metadata(entry, entry.get("bloom_level"))

    legacy_bloom_rows = db.list_bloom(user_id=user_id, limit=200)
    for row in legacy_bloom_rows:
        topic = row["topic"]
        if topic not in bloom_progress:
            bloom_progress[topic] = {
                "current_level": row["level"],
                "updated_at": row["updated_at"],
            }
            _attach_bloom_metadata(bloom_progress[topic], bloom_progress[topic].get("current_level"))
        entry = skills.setdefault(topic, {"theta": None, "p_know": None, "updated_at": row["updated_at"], "recent_attempts": []})
        entry.setdefault("bloom_level", row["level"])
        entry.setdefault("bloom_updated_at", row["updated_at"])
        _attach_bloom_metadata(entry, entry.get("bloom_level"))

    assessments = profile_batch.get("recent_assessments", [])
    attempts_by_domain: dict[str, list[dict[str, Any]]] = {}
    for item in assessments:
        domain = item.get("domain") or ""
        attempt_entry = {
            "assessment_id": item.get("assessment_id"),
            "item_id": item.get("item_id"),
            "score": item.get("score"),
            "confidence": item.get("confidence"),
            "source": item.get("source"),
            "bloom_level": item.get("bloom_level"),
            "self_assessment": item.get("self_assessment"),
            "when": item.get("created_at"),
        }
        _attach_bloom_metadata(attempt_entry, attempt_entry.get("bloom_level"))
        attempts_by_domain.setdefault(domain, []).append(attempt_entry)

    for domain, entries in attempts_by_domain.items():
        entries.sort(key=lambda e: e.get("when") or "", reverse=True)
        trimmed = entries[:5]
        skill_entry = skills.setdefault(domain, {"theta": None, "p_know": None, "updated_at": None, "recent_attempts": []})
        skill_entry["recent_attempts"] = trimmed
        if trimmed:
            skill_entry["last_evidence_id"] = trimmed[0]["assessment_id"]

    for entry in skills.values():
        if entry.get("p_know") is None and entry.get("theta") is not None:
            pk = 1 / (1 + math.exp(-entry["theta"]))
            entry["p_know"] = round(pk, 3)
        entry.setdefault("recent_attempts", [])
        _attach_bloom_metadata(entry, entry.get("bloom_level"))

    spaced_due = db.compute_spaced_reviews(user_id, limit=10)
    recommendations = db.list_recent_recommendations(user_id, limit=5)
    feedback_summary = db.aggregate_feedback(user_id=user_id)

    return {
        "user_id": user_id,
        "skills": skills,
        "bloom_progress": bloom_progress,
        "recent_attempts": assessments[:5],
        "spaced_due": spaced_due,
        "recommendations": recommendations,
        "feedback_summary": feedback_summary,
    }


# ---------- Learner Model ----------
@app.get("/learner/model/get", response_model=LearnerModel)
def learner_model_get(user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    db.ensure_user(user_id)
    return db.get_learner_model(user_id)


@app.post("/learner/model/update", response_model=LearnerModel)
def learner_model_update(model: LearnerModel):
    if not model.user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    db.ensure_user(model.user_id)
    try:
        updated = db.update_learner_model(model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        preferences_payload = updated.preferences.model_dump(
            mode="json", exclude_none=True
        )
        LEARNING_PATH_MANAGER.persist_preferences(
            updated.user_id,
            preferences_payload,
        )
    except Exception:
        logger.exception("Failed to persist learner preferences into learning path state")

    return updated


@app.post("/copilot/plan")
def copilot_plan(body: CopilotPlanRequest):
    db.ensure_user(body.teacher_id)
    objectives_text = "\n".join(f"- {obj}" for obj in body.objectives)
    constraint_text = f"\nConstraints: {body.constraints}" if body.constraints else ""
    prompt = (
        f"Design a concise lesson plan for topic: {body.topic}.\n"
        f"Objectives:\n{objectives_text}{constraint_text}\n"
        f"Ensure each section includes a Bloom level tag ({_BLOOM_RANGE_TEXT}) and actionable activity guidance."
    )
    messages = [
        {"role": "system", "content": COPILOT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        content = _llm_call(
            messages,
            max_tokens=1200,
            user_id=body.teacher_id,
            prompt_version="copilot.plan.v1",
            prompt_variant="copilot_plan",
            path_decisions=["copilot_plan", body.topic],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Copilot plan generation failed: {exc}")

    try:
        plan_model = parse_json_safe(content, CopilotPlanDraft)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Invalid plan JSON: {exc}")
    try:
        db.mark_last_llm_metric_validated(body.teacher_id)
    except Exception:
        pass

    plan_dict = plan_model.model_dump()
    if body.teacher_notes:
        plan_dict["teacher_notes"] = body.teacher_notes

    locale = body.locale or "en"
    culture_report = tutor.culture_sensitivity_check(json.dumps(plan_dict, ensure_ascii=False), locale)
    if culture_report.get("flags"):
        plan_dict["culture_flags"] = culture_report["flags"]

    bloom_alignment = [
        {"section": section.title, "bloom_level": section.bloom_level}
        for section in plan_model.sections
    ]
    provenance = {
        "model": tutor.MODEL_ID,
        "prompt_version": "copilot.plan.v1",
        "generated_at": datetime.utcnow().isoformat(),
        "objectives": body.objectives,
        "culture_check": culture_report,
    }

    saved = db.save_copilot_plan(
        teacher_id=body.teacher_id,
        topic=body.topic,
        objectives=body.objectives,
        plan=plan_dict,
        bloom_alignment=bloom_alignment,
        provenance=provenance,
        status="draft",
    )
    return {"plan": saved}


@app.post("/copilot/moderate")
def copilot_moderate(body: CopilotModerationBody):
    status_map = {
        "approved": "approved",
        "needs_revision": "revision_requested",
        "rejected": "rejected",
    }
    status = body.status or status_map.get(body.decision, body.decision)
    try:
        updated = db.record_copilot_moderation(
            plan_id=body.plan_id,
            moderator_id=body.moderator_id,
            decision=body.decision,
            rationale=body.rationale,
            flags=body.flags,
            status=status,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="plan not found")
    return {"plan": updated, "status": updated.get("status")}


def _run_eval_probe(probe: dict[str, Any], use_llm: bool) -> dict[str, Any]:
    expected = [k.lower() for k in probe.get("expected_keywords", [])]
    start = time.perf_counter()
    response = ""
    latency_ms = 0.0
    json_valid = False
    if use_llm:
        answer = generate_with_continue(
            system="You are a rigorous evaluator. Answer concisely.",
            user_text=probe["prompt"],
            max_tokens=probe.get("max_tokens"),
            user_id=None,
            prompt_version="eval.v1",
            path_decisions=[
                "eval_probe",
                str(probe.get("id") or probe.get("slug") or probe.get("name") or "unknown"),
            ],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        response = answer.strip()
        try:
            _extract_trailing_json(response)
            json_valid = True
        except Exception:
            json_valid = False
    else:
        latency_ms = 0.0
        response = "offline-eval " + " ".join(expected) if expected else "offline-eval"
        json_valid = True

    lower = response.lower()
    accuracy = 1.0 if expected and all(keyword in lower for keyword in expected) else (1.0 if not expected else 0.0)
    return {
        "accuracy": accuracy,
        "json_valid": json_valid,
        "latency_ms": round(latency_ms, 2),
        "used_llm": use_llm,
        "response_excerpt": response[:200],
    }


@app.post("/eval/run")
def eval_run(body: EvalRequest):
    run_id = body.run_id or _short_assessment_id("eval")
    use_llm = bool(body.use_llm or os.getenv("EVAL_USE_LLM", "false").lower() == "true")
    probes = EVAL_PROBES
    if body.sample_size:
        probes = probes[: max(1, min(len(EVAL_PROBES), int(body.sample_size)))]

    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    json_flags: list[bool] = []

    for probe in probes:
        metrics = _run_eval_probe(probe, use_llm)
        latencies.append(metrics.get("latency_ms", 0.0))
        json_flags.append(bool(metrics.get("json_valid")))
        db.record_eval_result(run_id, probe["id"], probe["category"], metrics)
        results.append({"probe": probe, "metrics": metrics})

    category_summary: dict[str, dict[str, Any]] = {}
    for item in results:
        cat = item["probe"]["category"]
        metrics = item["metrics"]
        entry = category_summary.setdefault(cat, {"count": 0, "accuracy_sum": 0.0})
        entry["count"] += 1
        entry["accuracy_sum"] += metrics.get("accuracy", 0.0)
    for entry in category_summary.values():
        if entry["count"]:
            entry["average_accuracy"] = round(entry["accuracy_sum"] / entry["count"], 3)

    overall_accuracy = (
        round(sum(item["metrics"].get("accuracy", 0.0) for item in results) / len(results), 3)
        if results
        else 0.0
    )
    json_valid_rate = (
        round(sum(1 for flag in json_flags if flag) / len(json_flags), 3)
        if json_flags
        else 0.0
    )
    median_latency = round(statistics.median(latencies), 2) if latencies else 0.0
    lat_sorted = sorted(latencies)
    p95_index = int(max(0, len(lat_sorted) - 1) * 0.95)
    p95_latency = round(lat_sorted[p95_index], 2) if lat_sorted else 0.0

    summary = {
        "run_id": run_id,
        "overall_accuracy": overall_accuracy,
        "json_valid_rate": json_valid_rate,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "category_breakdown": category_summary,
    }

    return {"summary": summary, "results": results}


def _validate_eval_session(session_id: Optional[str], learner_id: str, topic: str) -> Optional[dict[str, Any]]:
    if not session_id:
        return None
    session = journey_tracker.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    owner = session.get("user_id")
    if owner and owner != learner_id:
        raise HTTPException(status_code=403, detail="session does not belong to learner")
    subject = session.get("subject_id")
    if subject and subject != topic:
        raise HTTPException(status_code=400, detail="topic mismatch with session subject")
    return session


@app.post("/eval/pretest")
def eval_pretest(body: EvalAttemptBody):
    _validate_eval_session(body.session_id, body.learner_id, body.topic)
    try:
        attempt = db.record_pretest_attempt(
            learner_id=body.learner_id,
            topic=body.topic,
            score=body.score,
            max_score=body.max_score,
            attempt_id=body.attempt_id,
            strategy=body.strategy,
            metadata=body.metadata,
            session_id=body.session_id,
            instrument_id=body.instrument_id,
            instrument_version=body.instrument_version,
            instrument=body.instrument,
            attempted_at=body.attempted_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    instrument = attempt.get("instrument")
    if body.session_id and instrument:
        attachment_details = journey.enrich_event_metadata(
            {
                "stage": "pretest",
                "instrument_id": instrument.get("instrument_id"),
                "version": instrument.get("version"),
            },
            subject_id=body.topic,
            skill_id=body.topic,
            competency_id=body.topic,
            event_type="evaluation.instrument.attached",
            outcome="in_progress",
        )
        journey_tracker.record_event(
            body.learner_id,
            body.topic,
            "evaluation.instrument.attached",
            session_id=body.session_id,
            details=attachment_details,
            skill_id=body.topic,
            competency_id=body.topic,
            outcome="in_progress",
        )

    try:
        ratio = attempt["score"] / attempt["max_score"] if attempt.get("max_score") else None
    except Exception:
        ratio = None
    result_details = journey.enrich_event_metadata(
        {
            "attempt_id": attempt.get("attempt_id") or attempt.get("id"),
            "instrument_id": instrument.get("instrument_id") if instrument else attempt.get("instrument_id"),
            "instrument_version": instrument.get("version") if instrument else attempt.get("instrument_version"),
        },
        subject_id=body.topic,
        skill_id=body.topic,
        competency_id=body.topic,
        score=ratio if isinstance(ratio, (int, float)) else None,
        event_type="evaluation.pretest.recorded",
    )
    journey_tracker.record_event(
        body.learner_id,
        body.topic,
        "evaluation.pretest.recorded",
        session_id=body.session_id,
        score=ratio if isinstance(ratio, (int, float)) else None,
        details=result_details,
        skill_id=body.topic,
        competency_id=body.topic,
    )
    return {"attempt": attempt}


@app.post("/eval/posttest")
def eval_posttest(body: EvalAttemptBody):
    _validate_eval_session(body.session_id, body.learner_id, body.topic)
    try:
        attempt = db.record_posttest_attempt(
            learner_id=body.learner_id,
            topic=body.topic,
            score=body.score,
            max_score=body.max_score,
            attempt_id=body.attempt_id,
            strategy=body.strategy,
            metadata=body.metadata,
            session_id=body.session_id,
            instrument_id=body.instrument_id,
            instrument_version=body.instrument_version,
            instrument=body.instrument,
            attempted_at=body.attempted_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    instrument = attempt.get("instrument")
    if body.session_id and instrument:
        attachment_details = journey.enrich_event_metadata(
            {
                "stage": "posttest",
                "instrument_id": instrument.get("instrument_id"),
                "version": instrument.get("version"),
            },
            subject_id=body.topic,
            skill_id=body.topic,
            competency_id=body.topic,
            event_type="evaluation.instrument.attached",
            outcome="in_progress",
        )
        journey_tracker.record_event(
            body.learner_id,
            body.topic,
            "evaluation.instrument.attached",
            session_id=body.session_id,
            details=attachment_details,
            skill_id=body.topic,
            competency_id=body.topic,
            outcome="in_progress",
        )

    try:
        ratio = attempt["score"] / attempt["max_score"] if attempt.get("max_score") else None
    except Exception:
        ratio = None
    result_details = journey.enrich_event_metadata(
        {
            "attempt_id": attempt.get("attempt_id") or attempt.get("id"),
            "instrument_id": instrument.get("instrument_id") if instrument else attempt.get("instrument_id"),
            "instrument_version": instrument.get("version") if instrument else attempt.get("instrument_version"),
        },
        subject_id=body.topic,
        skill_id=body.topic,
        competency_id=body.topic,
        score=ratio if isinstance(ratio, (int, float)) else None,
        event_type="evaluation.posttest.recorded",
    )
    journey_tracker.record_event(
        body.learner_id,
        body.topic,
        "evaluation.posttest.recorded",
        session_id=body.session_id,
        score=ratio if isinstance(ratio, (int, float)) else None,
        details=result_details,
        skill_id=body.topic,
        competency_id=body.topic,
    )
    return {"attempt": attempt}


@app.get("/eval/report")
def eval_report_summary(
    learner_id: Optional[str] = None,
    topic: Optional[str] = None,
    strategy: Optional[str] = None,
):
    pairs = db.fetch_normalized_gains(
        learner_id=learner_id,
        topic=topic,
        strategy=strategy,
    )
    summary = db.summarize_normalized_gains(
        learner_id=learner_id,
        topic=topic,
        strategy=strategy,
    )
    pre_values = [item.get("pre_normalized") for item in pairs if item.get("pre_normalized") is not None]
    post_values = [item.get("post_normalized") for item in pairs if item.get("post_normalized") is not None]
    delta_values = [item.get("delta") for item in pairs if item.get("delta") is not None]
    valid_gains = [item["normalized_gain"] for item in pairs if item.get("normalized_gain") is not None]

    def _average(values: list[float]) -> Optional[float]:
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    overall = {
        "pair_count": len(pairs),
        "valid_gain_count": len(valid_gains),
        "mean_pre": _average(pre_values),
        "mean_post": _average(post_values),
        "mean_delta": _average(delta_values),
        "average_normalized_gain": _average(valid_gains),
        "gain_confidence_interval": db.mean_confidence_interval(valid_gains),
    }
    return {
        "filters": {"learner_id": learner_id, "topic": topic, "strategy": strategy},
        "overall": overall,
        "summary": summary,
        "pairs": pairs,
    }


@app.get("/eval/export")
def eval_export(run_id: Optional[str] = None, limit: int = 500):
    rows = db.list_eval_results(run_id=run_id, limit=limit)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["run_id", "probe_id", "category", "accuracy", "json_valid", "latency_ms", "response_excerpt", "created_at"])
    for row in rows:
        metrics = row.get("metrics") or {}
        writer.writerow(
            [
                row.get("run_id"),
                row.get("probe_id"),
                row.get("category"),
                metrics.get("accuracy"),
                metrics.get("json_valid"),
                metrics.get("latency_ms"),
                metrics.get("response_excerpt"),
                row.get("created_at"),
            ]
    )
    return Response(content=output.getvalue(), media_type="text/csv")

# ---------- Privacy ----------
@app.post("/privacy/consent")
def privacy_consent(body: PrivacyConsentBody):
    if not body.user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    if body.consented:
        db.ensure_user(body.user_id)
    try:
        db.record_privacy_consent(body.user_id, body.consented, body.consent_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"consent": db.get_privacy_consent(body.user_id)}


@app.get("/privacy/export")
def privacy_export(user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    bundle = db.export_user_data(user_id)
    if not bundle.get("users") and all(not bundle.get(key) for key in bundle if key not in {"user_id", "users"}):
        raise HTTPException(status_code=404, detail="user not found")
    return bundle


@app.delete("/privacy/delete")
def privacy_delete(user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")

    deleted = db.delete_user_data(user_id)
    return {"user_id": user_id, "deleted": deleted}


def _select_diagnostic_items(user_id: str, skill: Optional[str], limit: int = 3) -> list[dict]:
    return journey.select_calibration_items(
        skill,
        limit=limit,
        user_id=user_id,
        db_module=db,
        item_bank=tutor.ITEM_BANK,
        elo_engine=ELO_ENGINE,
    )


@app.post("/diagnose/start")
def diagnose_start(body: DiagnoseStartBody):
    db.ensure_user(body.user_id)
    items = _select_diagnostic_items(body.user_id, body.skill)
    db.log_journey_update(
        body.user_id,
        "diagnostic_start",
        {"skill": body.skill, "selected_items": [item["id"] for item in items]},
    )
    session = journey_tracker.start_session(
        user_id=body.user_id,
        subject_id=body.skill,
        session_type="diagnostic",
        metadata={"skill": body.skill},
    )
    selection_details = journey.enrich_event_metadata(
        {"item_ids": [item["id"] for item in items]},
        subject_id=body.skill,
        skill_id=body.skill,
        competency_id=body.skill,
        event_type="diagnostic_items_selected",
        outcome="assigned",
    )
    journey_tracker.record_event(
        user_id=body.user_id,
        subject_id=body.skill,
        event_type="diagnostic_items_selected",
        details=selection_details,
        session_id=session["session_id"],
        skill_id=body.skill,
        competency_id=body.skill,
        outcome="assigned",
    )
    return {"diagnostic_items": items, "journey_session": session}


@app.post("/journey/diagnostic/start")
def journey_diagnostic_start(body: JourneyDiagnosticStartBody):
    db.ensure_user(body.user_id)
    topic = body.subject_id or body.skill
    limit = body.limit or 3
    items = _select_diagnostic_items(body.user_id, topic, limit=limit)
    db.log_journey_update(
        body.user_id,
        "journey_diagnostic_start",
        {
            "subject_id": topic,
            "limit": limit,
            "selected_items": [item.get("id") for item in items],
        },
    )
    session = journey_tracker.start_session(
        user_id=body.user_id,
        subject_id=topic,
        session_type="diagnostic",
        metadata={"skill": topic, "mode": "calibration"},
    )
    calibration = journey.prepare_diagnostic_calibration(
        body.user_id,
        topic,
        db_module=db,
        elo_engine=ELO_ENGINE,
    )
    selection_details = journey.enrich_event_metadata(
        {"item_ids": [item.get("id") for item in items]},
        subject_id=topic,
        skill_id=topic,
        competency_id=topic,
        event_type="diagnostic_items_selected",
        outcome="assigned",
    )
    journey_tracker.record_event(
        user_id=body.user_id,
        subject_id=topic,
        event_type="diagnostic_items_selected",
        details=selection_details,
        session_id=session["session_id"],
        skill_id=topic,
        competency_id=topic,
        outcome="assigned",
    )
    calibration_details = journey.enrich_event_metadata(
        {"calibration": calibration},
        subject_id=topic,
        skill_id=topic,
        competency_id=topic,
        event_type="diagnostic_calibration",
        outcome="completed",
    )
    journey_tracker.record_event(
        user_id=body.user_id,
        subject_id=topic,
        event_type="diagnostic_calibration",
        details=calibration_details,
        session_id=session["session_id"],
        skill_id=topic,
        competency_id=topic,
        outcome="completed",
    )
    return {
        "diagnostic_items": items,
        "journey_session": session,
        "calibration": calibration,
    }


# ---------- Journey APIs ----------
@app.post("/journey/session/start")
def journey_session_start(body: JourneySessionStartBody):
    db.ensure_user(body.user_id)
    return journey_tracker.start_session(
        user_id=body.user_id,
        subject_id=body.subject_id,
        session_type=body.session_type,
        metadata=body.metadata,
    )


@app.post("/journey/session/end")
def journey_session_end(body: JourneySessionEndBody):
    session = journey_tracker.get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    if session.get("user_id") != body.user_id:
        raise HTTPException(status_code=403, detail="session does not belong to user")
    record = journey_tracker.complete_session(
        user_id=body.user_id,
        session_id=body.session_id,
        summary=body.summary,
        ended_at=body.ended_at,
    )
    if not record:
        raise HTTPException(status_code=404, detail="session not found")
    return record


@app.post("/journey/event")
def journey_event(body: JourneyEventBody):
    db.ensure_user(body.user_id)
    subject_id = body.subject_id
    if body.session_id:
        session = journey_tracker.get_session(body.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="session not found")
        if session.get("user_id") != body.user_id:
            raise HTTPException(status_code=403, detail="session does not belong to user")
        session_subject = session.get("subject_id")
        if subject_id and session_subject and subject_id != session_subject:
            raise HTTPException(status_code=400, detail="subject mismatch with session")
        if not subject_id:
            subject_id = session_subject
    enriched_details = journey.enrich_event_metadata(
        body.details,
        subject_id=subject_id,
        skill_id=body.skill_id,
        competency_id=body.competency_id,
        score=body.score,
        event_type=body.event_type,
        outcome=body.outcome or body.status,
        status=body.status,
    )
    if not (enriched_details.get("skill_id") or enriched_details.get("competency_id")):
        raise HTTPException(status_code=400, detail="skill metadata required for event")
    if not enriched_details.get("outcome"):
        raise HTTPException(status_code=400, detail="outcome status required for event")
    return journey_tracker.record_event(
        user_id=body.user_id,
        subject_id=subject_id,
        event_type=body.event_type,
        lesson_id=body.lesson_id,
        score=body.score,
        details=enriched_details,
        session_id=body.session_id,
        skill_id=body.skill_id,
        competency_id=body.competency_id,
        outcome=body.outcome,
        status=body.status,
    )


@app.get("/journey/timeline")
def journey_timeline(
    user_id: str,
    subject_id: Optional[str] = None,
    limit_sessions: int = 10,
    limit_events: int = 50,
):
    db.ensure_user(user_id)
    return journey_tracker.get_timeline(
        user_id=user_id,
        subject_id=subject_id,
        limit_sessions=limit_sessions,
        limit_events=limit_events,
    )


# ---------- Teacher analytics ----------


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _attach_gain_metrics(entry: dict[str, Any], pair: dict[str, Any]) -> None:
    pre_score = _coerce_float(pair.get("pre_score"))
    post_score = _coerce_float(pair.get("post_score"))
    entry["pre_score"] = pre_score
    entry["post_score"] = post_score
    entry["pre_max_score"] = _coerce_float(pair.get("pre_max_score"))
    entry["post_max_score"] = _coerce_float(pair.get("post_max_score"))
    entry["gain_strategy"] = pair.get("strategy")
    entry["pre_attempted_at"] = pair.get("pre_attempted_at")
    entry["post_attempted_at"] = pair.get("post_attempted_at")
    entry["pre_session_id"] = pair.get("pre_session_id")
    entry["post_session_id"] = pair.get("post_session_id")

    norm_gain = _coerce_float(pair.get("normalized_gain"))
    entry["normalized_gain"] = round(norm_gain, 4) if norm_gain is not None else None

    if pre_score is not None and post_score is not None:
        entry["score_delta"] = round(post_score - pre_score, 4)
    else:
        entry["score_delta"] = None


_FLAG_REASON_FIELDS = {
    "low_confidence": "flag_low_confidence",
    "high_hints": "flag_high_hints",
    "regression": "flag_regression",
}


def _normalize_flag_fields(entry: dict[str, Any]) -> None:
    reasons_raw = entry.get("flagged_reasons")
    if isinstance(reasons_raw, list):
        reasons = [str(reason) for reason in reasons_raw if reason]
    elif isinstance(reasons_raw, str):
        reasons = [str(reasons_raw)] if reasons_raw else []
    else:
        reasons = []
    seen: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    entry["flagged_reasons"] = seen
    for reason, field in _FLAG_REASON_FIELDS.items():
        entry[field] = bool(entry.get(field)) or (reason in seen)
    entry["stuck_flag"] = bool(entry.get("stuck_flag")) or bool(seen)


def _add_flag_reason(entry: dict[str, Any], reason: str) -> None:
    if not reason:
        return
    reasons = entry.get("flagged_reasons")
    if isinstance(reasons, list):
        normalized = [str(item) for item in reasons if item]
    elif isinstance(reasons, str):
        normalized = [str(reasons)] if reasons else []
    else:
        normalized = []
    if reason not in normalized:
        normalized.append(reason)
    entry["flagged_reasons"] = normalized
    _normalize_flag_fields(entry)


def _summarize_evidence(evidence: Any) -> Optional[str]:
    if evidence is None:
        return None
    if isinstance(evidence, (str, int, float, bool)):
        return str(evidence)
    if isinstance(evidence, dict):
        parts: list[str] = []
        for key, value in evidence.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}={value}")
            elif isinstance(value, dict):
                label = value.get("summary") or value.get("description") or value.get("id") or value.get("name")
                if label is not None:
                    parts.append(f"{key}={label}")
                else:
                    parts.append(f"{key}=dict")
            elif isinstance(value, list):
                parts.append(f"{key}={len(value)} items")
            else:
                parts.append(f"{key}={value}")
        return "; ".join(parts) if parts else None
    if isinstance(evidence, list):
        parts: list[str] = []
        for item in evidence:
            if isinstance(item, dict):
                label = item.get("summary") or item.get("description") or item.get("id") or item.get("name")
                if label:
                    parts.append(str(label))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False)[:40])
            else:
                parts.append(str(item))
            if len(parts) >= 5:
                break
        return ", ".join(parts) if parts else None
    return json.dumps(evidence, ensure_ascii=False)[:200]


def _load_path_events(user_id: Optional[str], subject_id: Optional[str]) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
    if not user_id:
        return [], None
    try:
        events = db.list_learning_path_events(user_id, subject_id)
    except Exception:
        logger.exception("Failed to load learning path events for analytics", extra={"user_id": user_id, "subject_id": subject_id})
        return [], None
    timeline = [dict(event) for event in reversed(events)]
    latest = timeline[-1] if timeline else None
    return timeline, latest


def _load_bloom_history(user_id: Optional[str], subject_id: Optional[str]) -> list[dict[str, Any]]:
    if not user_id or not subject_id:
        return []
    try:
        rows = db.list_bloom_progress_history(user_id=user_id, topic=subject_id, limit=200)
    except Exception:
        logger.exception(
            "Failed to load bloom history for analytics",
            extra={"user_id": user_id, "subject_id": subject_id},
        )
        return []
    history = [dict(row) for row in rows]
    history.sort(key=lambda item: item.get("created_at") or "")
    return history


@app.get("/teacher/analytics")
def teacher_analytics(
    subject_id: Optional[str] = None,
    only_flagged: bool = False,
    window_days: int = 7,
    limit: int = 100,
):
    try:
        days = max(1, min(90, int(window_days)))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="window_days must be an integer")
    snapshot = db.compute_teacher_analytics(window_days=days)
    for entry in snapshot:
        _normalize_flag_fields(entry)

    gain_pairs = db.fetch_normalized_gains(topic=subject_id if subject_id else None)
    gain_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in gain_pairs:
        learner = pair.get("learner_id")
        topic_key = pair.get("topic")
        if not learner or not topic_key:
            continue
        gain_lookup[(str(learner), str(topic_key))] = pair

    existing_keys: set[tuple[str, str]] = set()
    for entry in snapshot:
        learner = entry.get("user_id")
        topic_key = entry.get("subject_id") or entry.get("topic")
        if learner and topic_key:
            key = (str(learner), str(topic_key))
            existing_keys.add(key)
            entry["topic"] = topic_key
            pair = gain_lookup.get(key)
            if pair:
                _attach_gain_metrics(entry, pair)
                if entry.get("score_delta") is not None and entry["score_delta"] < -0.01:
                    _add_flag_reason(entry, "regression")
                if entry.get("normalized_gain") is not None and entry["normalized_gain"] < 0:
                    _add_flag_reason(entry, "regression")
        _normalize_flag_fields(entry)

    for key, pair in gain_lookup.items():
        if key in existing_keys:
            continue
        learner, topic_key = key
        new_entry: dict[str, Any] = {
            "user_id": learner,
            "subject_id": topic_key,
            "topic": topic_key,
            "window_days": days,
            "hint_count": 0,
            "low_confidence_count": 0,
            "history_tail": [],
            "stuck_flag": False,
            "flag_low_confidence": False,
            "flag_high_hints": False,
            "flag_regression": False,
            "flagged_reasons": [],
            "analytics_updated_at": None,
            "state_updated_at": None,
            "recent_confidence": None,
            "recent_score": None,
            "confidence_trend": None,
            "confidence_interval_lower": None,
            "confidence_interval_upper": None,
            "confidence_interval_mean": None,
            "confidence_interval_margin": None,
            "confidence_interval_width": None,
            "confidence_interval_confidence_level": None,
            "confidence_interval_sample_size": 0,
        }
        _attach_gain_metrics(new_entry, pair)
        if new_entry.get("score_delta") is not None and new_entry["score_delta"] < -0.01:
            _add_flag_reason(new_entry, "regression")
        if new_entry.get("normalized_gain") is not None and new_entry["normalized_gain"] < 0:
            _add_flag_reason(new_entry, "regression")
        _normalize_flag_fields(new_entry)
        snapshot.append(new_entry)

    results: list[dict[str, Any]] = []
    for entry in snapshot:
        if subject_id and entry.get("subject_id") != subject_id and entry.get("topic") != subject_id:
            continue
        if only_flagged and not entry.get("stuck_flag"):
            continue
        _normalize_flag_fields(entry)
        learner_id = entry.get("user_id")
        topic_key = entry.get("subject_id") or entry.get("topic")
        timeline, latest_event = _load_path_events(learner_id, topic_key)
        entry["learning_path_events"] = timeline
        entry["latest_path_event"] = latest_event
        entry["latest_path_rationale"] = latest_event.get("reason") if latest_event else None
        entry["latest_path_evidence"] = latest_event.get("evidence") if latest_event else None
        entry["latest_path_evidence_summary"] = _summarize_evidence(latest_event.get("evidence")) if latest_event else None
        entry["latest_path_confidence"] = latest_event.get("confidence") if latest_event else None
        entry["bloom_history"] = _load_bloom_history(learner_id, topic_key)
        if learner_id:
            entry["feedback_summary"] = db.aggregate_feedback(user_id=learner_id)
        else:
            entry["feedback_summary"] = {"total": 0, "ratings": {}, "average_confidence": None}
        results.append(entry)
        if limit and len(results) >= int(limit):
            break
    return results


@app.post("/teacher/learning-path/override")
def teacher_learning_path_override(body: LearningPathOverrideBody):
    db.ensure_user(body.user_id)
    window_days = body.window_days or 7
    state = db.apply_learning_path_override(
        body.user_id,
        body.subject_id,
        target_level=body.target_level,
        notes=body.notes,
        expires_at=body.expires_at,
        applied_by=body.applied_by,
        metadata=body.metadata,
    )

    bloom_level = (
        body.target_level
        or (state.get("current_level") if isinstance(state, dict) else None)
        or _LOWEST_BLOOM_LEVEL
    )
    evidence = {
        "applied_by": body.applied_by,
        "notes": body.notes,
        "metadata": body.metadata,
    }
    db.log_learning_path_event(
        body.user_id,
        body.subject_id,
        bloom_level=bloom_level,
        action="manual_override",
        reason=body.notes,
        reason_code="manual_override",
        confidence=None,
        evidence=evidence,
    )

    snapshot = db.compute_teacher_analytics(window_days=window_days)
    analytics_entry = next(
        (item for item in snapshot if item.get("user_id") == body.user_id and item.get("subject_id") == body.subject_id),
        None,
    )
    refreshed_state = db.get_learning_path_state(body.user_id, body.subject_id)
    return {"state": refreshed_state, "analytics": analytics_entry}


# ---------- DB Inspect ----------
@app.get("/db/mastery")
def db_mastery(user_id: Optional[str] = None, limit: int = 100):
    return [dict(r) for r in db.list_mastery(user_id, limit)]

@app.get("/db/prompts")
def db_prompts(topic: Optional[str] = None, limit: int = 50):
    return [dict(r) for r in db.list_prompts(topic, limit)]

@app.get("/db/journey")
def db_journey(user_id: Optional[str] = None, limit: int = 100):
    return db.list_journey(user_id, limit)

@app.get("/db/chat_ops")
def db_chat_ops(user_id: Optional[str] = None, limit: int = 100):
    return db.list_chat_ops(user_id, limit)

@app.get("/db/items")
def db_items(skill: Optional[str] = None, limit: int = 100):
    return [dict(r) for r in db.list_items(skill, limit)]

@app.get("/db/bloom")
def db_bloom(user_id: Optional[str] = None, limit: int = 100):
    return [dict(r) for r in db.list_bloom(user_id, limit)]

# ---------- Learning Interventions ----------
@app.post("/learning/intervention", response_model=InterventionResponse)
async def get_learning_intervention(request: Request, pattern: LearningPatternRequest):
    """Generate learning interventions based on the user's learning pattern."""
    try:
        user_id = request.state.user_id
        
        # Validate timestamps
        now = datetime.utcnow()
        if any(ts > now for ts in pattern.timestamps):
            raise HTTPException(
                status_code=400,
                detail="Invalid timestamps: cannot be in the future"
            )
        
        # Validate data ranges
        if any(not 0 <= score <= 1 for score in pattern.accuracy_scores):
            raise HTTPException(
                status_code=400,
                detail="Accuracy scores must be between 0 and 1"
            )
            
        if any(not 0 <= level <= 1 for level in pattern.engagement_levels):
            raise HTTPException(
                status_code=400,
                detail="Engagement levels must be between 0 and 1"
            )
            
        if any(time < 0 for time in pattern.response_times):
            raise HTTPException(
                status_code=400,
                detail="Response times must be non-negative"
            )
            
        user_profile = db.get_user_profile(user_id) or {}

        snapshot_payload = pattern.progress_snapshot if isinstance(pattern.progress_snapshot, dict) else None
        progress_snapshot: dict[str, Any] = copy.deepcopy(snapshot_payload) if snapshot_payload else {}
        objective_id = pattern.objective_id
        resolved_bloom = _clean_bloom_level(pattern.bloom_level)

        subject_id = (
            progress_snapshot.get("subject_id")
            or progress_snapshot.get("topic")
            or progress_snapshot.get("domain")
        )
        if not subject_id:
            subject_id = request.query_params.get("subject_id")
        if not subject_id:
            subject_id = (
                user_profile.get("subject_id")
                or user_profile.get("subject")
                or user_profile.get("topic")
            )
        if not subject_id:
            try:
                states = db.list_learning_path_states(user_id=user_id)
            except Exception:
                states = []
            for entry in states:
                candidate = entry.get("subject_id")
                if candidate and candidate != "__global__":
                    subject_id = candidate
                    break

        canonical_subject = _canonical_domain(subject_id) if subject_id else None

        try:
            learning_state = LEARNING_PATH_MANAGER.get_state(user_id, subject_id) if subject_id else {}
        except Exception:
            learning_state = {}
        if not isinstance(learning_state, dict):
            learning_state = {}

        levels_raw = learning_state.get("levels") if isinstance(learning_state.get("levels"), dict) else {}
        normalized_levels = {lvl: float(val) for lvl, val in levels_raw.items() if isinstance(lvl, str)}
        state_level = _clean_bloom_level(learning_state.get("current_level"))
        resolved_bloom = resolved_bloom or state_level or _LOWEST_BLOOM_LEVEL

        if subject_id:
            progress_snapshot.setdefault("subject_id", subject_id)
        if canonical_subject:
            progress_snapshot.setdefault("domain", canonical_subject)
        progress_snapshot.setdefault("levels", normalized_levels)
        progress_snapshot.setdefault("current_level", resolved_bloom)
        if learning_state.get("updated_at") and "updated_at" not in progress_snapshot:
            progress_snapshot["updated_at"] = learning_state["updated_at"]

        preferences = learning_state.get("preferences") if isinstance(learning_state.get("preferences"), dict) else {}
        mastered_nodes: set[str] = set()
        if canonical_subject:
            mastered_nodes = _mastered_nodes_for_domain(canonical_subject, normalized_levels, resolved_bloom)
            if mastered_nodes:
                progress_snapshot.setdefault("mastered_nodes", sorted(mastered_nodes))

        if canonical_subject:
            target_skill = _select_skill_definition(canonical_subject, resolved_bloom)
            if target_skill:
                objective_id = objective_id or target_skill.skill_id
                progress_snapshot.setdefault("target_skill_id", target_skill.skill_id)
                progress_snapshot.setdefault("target_skill_label", target_skill.label)
                progress_snapshot.setdefault("target_bloom", target_skill.bloom_level)
                try:
                    plan = _plan_learning_path(
                        user_id,
                        subject_id or canonical_subject,
                        canonical_subject,
                        mastered_nodes=mastered_nodes,
                        target_skill_id=target_skill.skill_id,
                        target_bloom=target_skill.bloom_level,
                        preferences=preferences,
                    )
                except Exception as exc:
                    logger.debug("Unable to build learning plan for interventions: %s", exc)
                    plan = None
                if isinstance(plan, dict):
                    ordered_nodes = plan.get("ordered_nodes") or []
                    if ordered_nodes:
                        first_node = ordered_nodes[0]
                        if isinstance(first_node, dict):
                            if first_node.get("id") and "recommended_objective" not in progress_snapshot:
                                progress_snapshot["recommended_objective"] = first_node.get("id")
                            if first_node.get("label") and "recommended_objective_label" not in progress_snapshot:
                                progress_snapshot["recommended_objective_label"] = first_node.get("label")
                            if objective_id is None and first_node.get("id"):
                                objective_id = first_node.get("id")
                    plan_preview = {
                        "next_nodes": ordered_nodes[:3],
                        "insights": plan.get("insights"),
                    }
                    if (plan_preview["next_nodes"] or plan_preview["insights"]) and "plan_preview" not in progress_snapshot:
                        progress_snapshot["plan_preview"] = plan_preview
                    skill_success = plan.get("skill_success")
                    if skill_success and "skill_success" not in progress_snapshot:
                        progress_snapshot["skill_success"] = skill_success

        progress_snapshot.setdefault("bloom_level", resolved_bloom)
        final_snapshot = progress_snapshot or None

        # Create learning pattern from request
        pattern_obj = LearningPattern(
            response_times=pattern.response_times,
            accuracy_scores=pattern.accuracy_scores,
            engagement_levels=pattern.engagement_levels,
            hint_usage=pattern.hint_usage,
            timestamps=pattern.timestamps,
            objective_id=objective_id,
            bloom_level=resolved_bloom,
            progress_snapshot=final_snapshot,
        )
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in intervention endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
    # Monitor for intervention triggers
    try:
        trigger = _INTERVENTION_SYSTEM.monitor_progress(user_id, pattern_obj)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not trigger:
        raise HTTPException(status_code=404, detail="No intervention needed at this time")
        
    # Generate intervention
    intervention = _INTERVENTION_SYSTEM.generate_intervention(trigger, user_profile)
    
    # Log intervention for analytics
    db.log_intervention(
        user_id=user_id,
        intervention_type=trigger.type,
        confidence=trigger.confidence,
        context=trigger.context,
        intervention=intervention
    )
    
    return InterventionResponse(
        type=trigger.type,
        confidence=trigger.confidence,
        detected_at=trigger.detected_at,
        context=trigger.context,
        intervention=intervention
    )

