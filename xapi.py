"""Utility helpers for emitting local xAPI statements with optional LRS forwarding.

The module defines a lightweight xAPI profile that captures the statements emitted by
AILearnBuddy. Statements are validated against this profile before they are persisted or
forwarded to an external Learning Record Store (LRS). Forwarding happens asynchronously
with retry/backoff to satisfy governance requirements around human oversight and
traceability.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

import requests

from db import DB_PATH

LOGGER = logging.getLogger("ailb.xapi")

# ---------------------------------------------------------------------------
# xAPI profile definition
# ---------------------------------------------------------------------------

XAPI_PROFILE_VERBS: dict[str, dict[str, str]] = {
    "http://adlnet.gov/expapi/verbs/answered": {
        "display": "answered",
        "description": "Learner submitted an answer to an activity or assessment item.",
    },
    "http://adlnet.gov/expapi/verbs/evaluated": {
        "display": "evaluated",
        "description": "System evaluated a learner submission and produced a score.",
    },
    "http://adlnet.gov/expapi/verbs/experienced": {
        "display": "experienced",
        "description": "Learner encountered or viewed a resource in the journey.",
    },
    "http://adlnet.gov/expapi/verbs/initialized": {
        "display": "initialized",
        "description": "Learner session or activity was initialised.",
    },
    "http://adlnet.gov/expapi/verbs/mastered": {
        "display": "mastered",
        "description": "Learner demonstrated mastery for a Bloom level or skill.",
    },
    "http://adlnet.gov/expapi/verbs/terminated": {
        "display": "terminated",
        "description": "Learner session or activity was completed or exited.",
    },
}

_ALLOWED_OBJECT_PREFIXES: Sequence[str] = (
    "activity:",
    "assessment:",
    "journey:",
    "https://",
    "http://",
    "urn:",
)

_ALLOWED_VERB_IDS: tuple[str, ...] = tuple(XAPI_PROFILE_VERBS.keys())
_ALLOWED_OBJECT_PREFIX_TEXT = ", ".join(_ALLOWED_OBJECT_PREFIXES)

_CONTEXT_EXTENSION_SCHEMA: dict[str, type] = {
    "bloom": str,
    "skill": str,
    "confidence": float,
    "path": str,
    "reason": str,
    "topic": str,
    "decision": str,
    "source": str,
    "window_days": int,
    "model_version": str,
    "session_type": str,
    "subject": str,
    "metadata": dict,
    "summary": dict,
    "event_type": str,
    "lesson_id": str,
    "session_id": str,
    "details": dict,
}

# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _to_bool_flag(value: Optional[bool]) -> Optional[int]:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _normalise_context(context: Optional[Dict[str, Any]]) -> Optional[str]:
    if context is None:
        return None
    try:
        return json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return None


def _coerce_extension(key: str, value: Any) -> Any:
    expected = _CONTEXT_EXTENSION_SCHEMA.get(key)
    if expected is None:
        raise ValueError(f"Unsupported context extension: {key}")
    if value is None:
        return None
    if expected is float:
        coerced = float(value)
        if key == "confidence" and not 0.0 <= coerced <= 1.0:
            raise ValueError("confidence extension must be between 0 and 1")
        return round(coerced, 6)
    if expected is int:
        return int(value)
    if expected is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{key} extension must be an object")
        return value
    return str(value)


def validate_statement(statement: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalise an xAPI statement according to the local profile."""

    if not isinstance(statement, dict):
        raise ValueError("statement must be a dict")

    actor = statement.get("actor")
    if not isinstance(actor, dict):
        raise ValueError("actor must be provided")
    account = actor.get("account") if isinstance(actor.get("account"), dict) else None
    if not account or not isinstance(account.get("name"), str) or not account.get("name").strip():
        raise ValueError("actor.account.name is required")
    if not isinstance(account.get("homePage"), str) or not account["homePage"].strip():
        raise ValueError("actor.account.homePage is required")

    verb = statement.get("verb")
    if not isinstance(verb, dict) or not isinstance(verb.get("id"), str):
        raise ValueError("verb.id must be provided")
    verb_id = verb["id"].strip()
    if not verb_id:
        raise ValueError("verb.id must be a non-empty string")
    verb["id"] = verb_id
    if verb_id not in XAPI_PROFILE_VERBS:
        allowed = ", ".join(sorted(_ALLOWED_VERB_IDS))
        raise ValueError(f"Unsupported verb '{verb_id}'. Allowed verbs: {allowed}")

    obj = statement.get("object")
    if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
        raise ValueError("object.id must be provided")
    object_id = obj["id"].strip()
    if not object_id:
        raise ValueError("object.id must be a non-empty string")
    obj["id"] = object_id
    if not any(object_id.startswith(prefix) for prefix in _ALLOWED_OBJECT_PREFIXES):
        raise ValueError(
            "object.id must start with one of the allowed prefixes: "
            f"{_ALLOWED_OBJECT_PREFIX_TEXT}"
        )

    result = statement.get("result")
    if result is not None:
        if not isinstance(result, dict):
            raise ValueError("result must be a dict when provided")
        score = result.get("score")
        if score is not None:
            if not isinstance(score, dict) or "raw" not in score:
                raise ValueError("result.score.raw is required when score is provided")
            raw_value = float(score["raw"])
            score["raw"] = raw_value
        if "success" in result:
            result["success"] = bool(result["success"])

    context = statement.get("context") or {}
    if not isinstance(context, dict):
        raise ValueError("context must be a dict")
    platform = context.get("platform") or os.getenv("XAPI_PLATFORM", "AILearnBuddy")
    language = context.get("language") or os.getenv("XAPI_LANGUAGE", "en")
    extensions = context.get("extensions") or {}
    if not isinstance(extensions, dict):
        raise ValueError("context.extensions must be a dict")

    cleaned_extensions: dict[str, Any] = {}
    for key, value in extensions.items():
        if key not in _CONTEXT_EXTENSION_SCHEMA:
            LOGGER.debug("Dropping unsupported xAPI extension: %s", key)
            continue
        cleaned = _coerce_extension(key, value)
        if cleaned is not None:
            cleaned_extensions[key] = cleaned

    context["platform"] = platform
    context["language"] = language
    context["extensions"] = cleaned_extensions
    statement["context"] = context

    return statement


async def _forward_statement_with_retry(
    statement: Dict[str, Any],
    *,
    lrs_url: str,
    headers: Dict[str, str],
    timeout: float = 5.0,
    max_attempts: int = 3,
) -> None:
    """Forward a statement to the configured LRS with exponential backoff."""

    delay = 0.5
    for attempt in range(1, max_attempts + 1):
        try:
            response = await asyncio.to_thread(
                requests.post,
                lrs_url,
                json=statement,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code < 500:
                return
            LOGGER.warning(
                "LRS responded with status %s on attempt %s", response.status_code, attempt
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to forward xAPI statement (attempt %s): %s", attempt, exc)
        if attempt == max_attempts:
            break
        await asyncio.sleep(delay)
        delay *= 2


def _schedule_forward(statement: Dict[str, Any], *, lrs_url: str, headers: Dict[str, str]) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = _forward_statement_with_retry(statement, lrs_url=lrs_url, headers=headers)

    if loop and loop.is_running():
        loop.create_task(coro)
    else:
        threading.Thread(target=lambda: asyncio.run(coro), daemon=True).start()


def emit(
    user_id: str,
    verb: str,
    object_id: str,
    *,
    score: Optional[float] = None,
    success: Optional[bool] = None,
    response: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the xAPI statement locally and forward to an LRS when configured."""

    stored_response: Optional[Any] = response
    if isinstance(response, (dict, list)):
        stored_response = json.dumps(response, ensure_ascii=False, separators=(",", ":"))

    payload: Dict[str, Any] = {
        "actor": {
            "account": {
                "homePage": os.getenv("APP_BASE_URL", "https://local.learning"),
                "name": user_id,
            }
        },
        "verb": {"id": verb},
        "object": {"id": object_id},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if score is not None or success is not None or response is not None:
        result_block: Dict[str, Any] = {}
        if score is not None:
            result_block["score"] = {"raw": float(score)}
        if success is not None:
            result_block["success"] = bool(success)
        if response is not None:
            result_block["response"] = response
        payload["result"] = result_block

    platform = os.getenv("XAPI_PLATFORM", "AILearnBuddy")
    base_context = {
        "platform": platform,
        "language": os.getenv("XAPI_LANGUAGE", "en"),
        "extensions": context or {},
    }
    payload["context"] = base_context

    validated = validate_statement(payload)
    validated_result = validated.get("result") or {}
    validated_extensions = (validated.get("context") or {}).get("extensions")

    context_json = _normalise_context(validated_extensions)

    with _connect() as con:
        con.execute(
            """
            INSERT INTO xapi_statements(user_id, verb, object_id, score, success, response, context)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                user_id,
                validated["verb"]["id"],
                validated["object"]["id"],
                None
                if "score" not in validated_result
                else float(validated_result["score"]["raw"]),
                _to_bool_flag(validated_result.get("success")),
                stored_response,
                context_json,
            ),
        )
        con.commit()

    lrs_url = os.getenv("LRS_URL")
    if not lrs_url:
        return

    headers = {
        "Content-Type": "application/json",
        "X-Experience-API-Version": "1.0.3",
    }
    auth = os.getenv("LRS_AUTH")
    if auth:
        headers["Authorization"] = auth

    _schedule_forward(validated, lrs_url=lrs_url, headers=headers)


