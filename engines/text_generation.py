"""Hybrid text generation and progression orchestration engine.

This module creates deterministic language-learning materials that map
vocabulary, grammar targets, and Bloom/HSK or CEFR levels. It also
integrates with the existing progression engine to close the loop between
content delivery, learner responses, and progression updates.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import db
from engines.progression import ProgressionEngine, ProgressionResult, ensure_progress_record

LexiconMode = Literal["simple", "broad"]


class UnknownSkillLevelError(ValueError):
    """Raised when a requested level is missing from the library."""


class UnknownLanguagePairError(ValueError):
    """Raised when a requested language pair is unavailable."""


_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "language"
_HSK_PATH = _DATA_ROOT / "hsk_levels.json"
_CEFR_PATH = _DATA_ROOT / "cefr_de_en_templates.json"


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_language_strategies() -> Dict[str, Dict[str, object]]:
    strategies: Dict[str, Dict[str, object]] = {}

    if _HSK_PATH.exists():
        hsk_payload = _load_json(_HSK_PATH)
        strategies["zh_en"] = {
            "strategy": "hsk",
            "levels": hsk_payload.get("levels", {}),
            "metadata": {"source": str(_HSK_PATH)},
        }

    if _CEFR_PATH.exists():
        cefr_payload = _load_json(_CEFR_PATH)
        strategies["de_en"] = {
            "strategy": "cefr",
            "levels": cefr_payload.get("levels", {}),
            "metadata": {"source": str(_CEFR_PATH)},
        }

    if not strategies:
        raise RuntimeError("No language strategies available; check data files.")

    return strategies


_LANGUAGE_STRATEGIES = _build_language_strategies()
LANGUAGE_LEVEL_LIBRARY: Dict[str, Dict[str, object]] = (
    _LANGUAGE_STRATEGIES.get("zh_en", {}).get("levels", {})  # type: ignore[assignment]
)


def _tokens_to_text(tokens: List[str]) -> str:
    """Join tokens into a readable string preserving punctuation spacing."""

    text = ""
    for token in tokens:
        if not text:
            text = token
            continue
        if token in {"，", "。", "！", "？", "；", ",", "."}:
            text += token
        else:
            text += token if text.endswith(" ") else f" {token}"
    return text


def _select_language_level(
    language_pair: str, skill_level: str
) -> Tuple[str, Dict[str, object], Dict[str, object]]:
    strategy = _LANGUAGE_STRATEGIES.get(language_pair)
    if not strategy:
        raise UnknownLanguagePairError(f"Unsupported language pair: {language_pair}")

    levels: Dict[str, object] = strategy.get("levels", {})  # type: ignore[assignment]
    if skill_level not in levels:
        raise UnknownSkillLevelError(f"Unsupported skill level: {skill_level}")

    return strategy["strategy"], levels[skill_level], strategy


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _token_overlap(reference: str, hypothesis: str) -> float:
    ref_tokens = set(_normalize_text(reference).split())
    hyp_tokens = set(_normalize_text(hypothesis).split())
    if not ref_tokens:
        return 0.0
    intersection = ref_tokens.intersection(hyp_tokens)
    return len(intersection) / len(ref_tokens)


class TextGenerationProgressionEngine:
    """Create level-aware language content and adapt progression."""

    def __init__(
        self,
        progression_engine: Optional[ProgressionEngine] = None,
    ) -> None:
        self.progression_engine = progression_engine or ProgressionEngine()
        self._interaction_log: List[Dict[str, object]] = []
        self._activity_cache: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    # Content generation
    # ------------------------------------------------------------------
    def generate_activity(
        self,
        user_id: str,
        subject_id: str,
        skill_level: str,
        lexicon_mode: LexiconMode = "simple",
        bloom_level: Optional[str] = None,
        language_pair: str = "zh_en",
    ) -> Dict[str, object]:
        if lexicon_mode not in ("simple", "broad"):
            raise ValueError("lexicon_mode must be 'simple' or 'broad'")

        strategy_name, level_config, strategy_meta = _select_language_level(
            language_pair, skill_level
        )
        lex_config = level_config.get(lexicon_mode)
        if not isinstance(lex_config, dict):
            raise ValueError(
                f"Lexicon mode '{lexicon_mode}' not available for {skill_level}"
            )

        ensure_progress_record(user_id, subject_id)

        text = _tokens_to_text(list(lex_config.get("text_tokens", [])))
        word_assignments = list(lex_config.get("vocab", []))
        exercises = list(lex_config.get("exercises", []))
        tasks = list(lex_config.get("tasks", []))

        assigned_bloom = bloom_level or level_config.get("default_bloom")
        activity_id = f"{language_pair}-{skill_level.lower()}-{lexicon_mode}-{uuid.uuid4().hex[:8]}"

        prompt_suite = dict(level_config.get("prompts", {}))

        payload = {
            "activity_id": activity_id,
            "skill_level": skill_level,
            "bloom_level": assigned_bloom,
            "lexicon_mode": lexicon_mode,
            "text": text,
            "word_assignments": word_assignments,
            "exercises": exercises,
            "tasks": tasks,
            "grammar_focus": level_config.get("grammar_focus", []),
            "competencies": level_config.get("competencies", []),
            "prompt_suite": prompt_suite,
            "language_pair": language_pair,
            "strategy": strategy_name,
            "strategy_metadata": strategy_meta.get("metadata", {}),
        }

        db.log_learning_event(
            user_id,
            subject_id,
            event_type="text_generation",
            details={
                "activity_id": activity_id,
                "skill_level": skill_level,
                "lexicon_mode": lexicon_mode,
                "bloom_level": assigned_bloom,
                "text": text,
                "language_pair": language_pair,
                "strategy": strategy_name,
            },
            skill_id=f"{language_pair}:{skill_level}:generation",
        )
        self._activity_cache[activity_id] = payload
        self._interaction_log.append(
            {
                "type": "generation",
                "activity_id": activity_id,
                "skill_level": skill_level,
                "lexicon_mode": lexicon_mode,
                "language_pair": language_pair,
                "strategy": strategy_name,
            }
        )
        return payload

    # ------------------------------------------------------------------
    # Response evaluation & progression update
    # ------------------------------------------------------------------
    def evaluate_response(
        self,
        user_id: str,
        subject_id: str,
        activity_id: str,
        score: float,
        response_text: Optional[str] = None,
        lexical_errors: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        ensure_progress_record(user_id, subject_id)
        lexical_errors = lexical_errors or []

        activity_meta = self._activity_cache.get(activity_id, {})
        lexicon_mode = activity_meta.get("lexicon_mode", "simple")
        language_pair = activity_meta.get("language_pair", "zh_en")
        skill_level = activity_meta.get("skill_level", "")

        progression_result: ProgressionResult = self.progression_engine.process_attempt(
            user_id=user_id,
            subject_id=subject_id,
            activity_id=activity_id,
            score=score,
            max_score=1.0,
        )

        feedback = self._craft_feedback(score, lexical_errors, str(lexicon_mode))
        next_steps = self._next_steps(
            score,
            lexical_errors,
            progression_result.new_level,
            current_mode=str(lexicon_mode),
        )

        event_details = {
            "activity_id": activity_id,
            "score": score,
            "lexical_errors": lexical_errors,
            "feedback": feedback,
            "next_steps": next_steps,
            "progression": asdict(progression_result),
            "language_pair": language_pair,
            "skill_level": skill_level,
        }
        if response_text:
            event_details["response_text"] = response_text

        db.log_learning_event(
            user_id,
            subject_id,
            event_type="text_response",
            details=event_details,
            score=score,
            skill_id=f"{language_pair}:{skill_level}:response" if skill_level else None,
        )
        self._interaction_log.append(
            {
                "type": "response",
                "activity_id": activity_id,
                "score": score,
                "lexical_errors": list(lexical_errors),
                "progression": asdict(progression_result),
                "language_pair": language_pair,
            }
        )

        return {
            "feedback": feedback,
            "next_steps": next_steps,
            "progression": progression_result,
        }

    # ------------------------------------------------------------------
    # Assessment pipelines
    # ------------------------------------------------------------------
    def assess_listening_comprehension(
        self,
        user_id: str,
        subject_id: str,
        activity_id: str,
        learner_response: str,
    ) -> Dict[str, object]:
        ensure_progress_record(user_id, subject_id)
        activity_meta = self._activity_cache.get(activity_id, {})
        prompt = (
            activity_meta.get("prompt_suite", {}).get("listening_comprehension", {})
        )
        language_pair = activity_meta.get("language_pair", "zh_en")
        skill_level = activity_meta.get("skill_level", "")

        keywords: List[str] = list(prompt.get("keywords", []))
        hits = sum(1 for kw in keywords if kw in learner_response)
        total_keywords = max(len(keywords), 1)
        score = hits / total_keywords

        audio_meta = dict(prompt.get("audio", {}))
        details = {
            "activity_id": activity_id,
            "assessment_type": "listening",
            "keywords": keywords,
            "learner_response": learner_response,
            "keyword_hits": hits,
            "total_keywords": total_keywords,
            "audio_metadata": audio_meta,
        }

        db.log_learning_event(
            user_id,
            subject_id,
            event_type="listening_assessment",
            details=details,
            score=score,
            skill_id=f"{language_pair}:{skill_level}:listening" if skill_level else None,
        )
        self._interaction_log.append(
            {
                "type": "listening_assessment",
                "activity_id": activity_id,
                "score": score,
                "keywords": keywords,
            }
        )
        return {
            "score": score,
            "keyword_hits": hits,
            "total_keywords": total_keywords,
            "audio": audio_meta,
        }

    def assess_pronunciation(
        self,
        user_id: str,
        subject_id: str,
        activity_id: str,
        audio_metadata: Dict[str, float],
    ) -> Dict[str, object]:
        ensure_progress_record(user_id, subject_id)
        activity_meta = self._activity_cache.get(activity_id, {})
        language_pair = activity_meta.get("language_pair", "zh_en")
        skill_level = activity_meta.get("skill_level", "")
        prompt = activity_meta.get("prompt_suite", {}).get("pronunciation", {})

        clarity = float(audio_metadata.get("clarity", 0.0))
        phoneme_accuracy = float(audio_metadata.get("phoneme_accuracy", 0.0))
        pacing = float(audio_metadata.get("pacing", 0.0))
        weighted_score = max(
            0.0,
            min(1.0, (clarity * 0.4) + (phoneme_accuracy * 0.4) + (pacing * 0.2)),
        )

        details = {
            "activity_id": activity_id,
            "assessment_type": "pronunciation",
            "audio_metadata": audio_metadata,
            "prompt_targets": prompt,
        }

        db.log_learning_event(
            user_id,
            subject_id,
            event_type="pronunciation_assessment",
            details=details,
            score=weighted_score,
            skill_id=f"{language_pair}:{skill_level}:pronunciation" if skill_level else None,
        )
        self._interaction_log.append(
            {
                "type": "pronunciation_assessment",
                "activity_id": activity_id,
                "score": weighted_score,
                "audio_metadata": audio_metadata,
            }
        )
        return {
            "score": weighted_score,
            "clarity": clarity,
            "phoneme_accuracy": phoneme_accuracy,
            "pacing": pacing,
        }

    def assess_translation(
        self,
        user_id: str,
        subject_id: str,
        activity_id: str,
        learner_translation: str,
    ) -> Dict[str, object]:
        ensure_progress_record(user_id, subject_id)
        activity_meta = self._activity_cache.get(activity_id, {})
        language_pair = activity_meta.get("language_pair", "zh_en")
        skill_level = activity_meta.get("skill_level", "")
        prompt = activity_meta.get("prompt_suite", {}).get("translation", {})

        reference = str(prompt.get("reference_translation", ""))
        overlap = _token_overlap(reference, learner_translation)
        exact_match = _normalize_text(reference) == _normalize_text(learner_translation)
        rubric = prompt.get("rubric")

        score = overlap if not exact_match else 1.0

        details = {
            "activity_id": activity_id,
            "assessment_type": "translation",
            "learner_translation": learner_translation,
            "reference_translation": reference,
            "lexical_overlap": overlap,
            "rubric": rubric,
            "exact_match": exact_match,
        }

        db.log_learning_event(
            user_id,
            subject_id,
            event_type="translation_assessment",
            details=details,
            score=score,
            skill_id=f"{language_pair}:{skill_level}:translation" if skill_level else None,
        )
        self._interaction_log.append(
            {
                "type": "translation_assessment",
                "activity_id": activity_id,
                "score": score,
                "overlap": overlap,
                "exact_match": exact_match,
            }
        )
        return {
            "score": score,
            "lexical_overlap": overlap,
            "exact_match": exact_match,
            "reference": reference,
        }

    # ------------------------------------------------------------------
    # Experimentation utilities
    # ------------------------------------------------------------------
    def suggest_experiments(self) -> List[Dict[str, object]]:
        return [
            {
                "name": "lexicon_mode_ab_test",
                "hypothesis": "Broader vocabulary exposures accelerate transfer once learners reach 0.75 accuracy.",
                "variants": {
                    "simple": {
                        "description": "Stay within current HSK/CEFR vocabulary and grammar targets.",
                        "trigger": "score < 0.75 or progression unchanged",
                    },
                    "broad": {
                        "description": "Inject higher-level vocabulary with explicit annotations.",
                        "trigger": "score >= 0.75 for two consecutive attempts",
                    },
                },
                "metrics": [
                    "average_score",
                    "progression.changed",
                    "lexical_error_rate",
                ],
                "status": "planned",
            },
            {
                "name": "feedback_tone_calibration",
                "hypothesis": "Specific correction categories reduce repeat lexical errors by 15%.",
                "variants": {
                    "direct": "Explicitly name tone/grammar issues in the first sentence.",
                    "scaffolded": "Ask guiding questions before presenting the correction.",
                },
                "metrics": ["lexical_error_recurrence", "learner_confidence"],
                "status": "planned",
            },
        ]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def interaction_log(self) -> List[Dict[str, object]]:
        return list(self._interaction_log)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _craft_feedback(
        self,
        score: float,
        lexical_errors: List[str],
        lexicon_mode: str,
    ) -> str:
        level_summary = "balanced" if lexicon_mode == "broad" else "core"
        if score >= 0.85:
            base = f"Excellent work! You handled the {level_summary} vocabulary with confidence."
        elif score >= 0.6:
            base = "Solid progress—let's reinforce a few points."
        else:
            base = "Let's revisit the key phrases together."

        if lexical_errors:
            issues = ", ".join(sorted(set(lexical_errors)))
            base += f" Focus on correcting: {issues}."
        else:
            base += " Continue extending your sentences with connectors."
        return base

    def _next_steps(
        self,
        score: float,
        lexical_errors: List[str],
        new_level: str,
        current_mode: str,
    ) -> List[Dict[str, str]]:
        actions: List[Dict[str, str]] = []
        if score >= 0.85:
            actions.append(
                {
                    "action": "Advance to scenario role-play",
                    "recommended_mode": "broad" if current_mode == "simple" else current_mode,
                    "target_level": new_level,
                }
            )
        else:
            actions.append(
                {
                    "action": "Repeat focused practice",
                    "recommended_mode": "simple",
                    "target_level": new_level,
                }
            )

        for category in sorted(set(lexical_errors)):
            if category:
                actions.append(
                    {
                        "action": f"Micro-drill on {category}",
                        "recommended_mode": "simple",
                        "target_level": new_level,
                    }
                )
        return actions


__all__ = [
    "TextGenerationProgressionEngine",
    "LANGUAGE_LEVEL_LIBRARY",
    "UnknownSkillLevelError",
    "UnknownLanguagePairError",
]
