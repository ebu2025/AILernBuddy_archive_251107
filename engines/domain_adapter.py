"""Domain adaptive orchestration for multimodal tutoring pipelines."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import ast
import math
import re
import statistics
import xml.etree.ElementTree as ET

import db

from engines.competency import (
    ALL_TEMPLATES,
    CompetencyTemplate,
    Recommendation,
    SkillDefinition,
    build_knowledge_graph,
)
from engines.progression import ProgressionEngine
from knowledge_graph import KnowledgeGraph, CompetencyNode
from schemas import (
    AssessmentErrorPattern,
    AssessmentResult,
    AssessmentStepEvaluation,
    RubricCriterion,
)

_DOMAIN_ALIASES: Dict[str, str] = {
    "bpmn": "business_process",
    "business": "business_process",
    "business_process": "business_process",
    "process": "business_process",
    "math": "mathematics",
    "mathematics": "mathematics",
    "language": "language",
    "language_de_en": "language",
    "language_zh_en": "language",
    "german": "language",
    "mandarin": "language",
    "chinese": "language",
}

_MEDIA_PRIORITIES: Dict[str, Sequence[str]] = {
    "language": ("audio", "dialogue", "transcript", "text"),
    "business_process": ("diagram_tools", "simulation", "notebook", "text"),
    "mathematics": ("plotter", "notebook", "visual", "text"),
}

_ASSESSMENT_MODEL_VERSION = "domain-adapter.v1"
_PROMPT_VERSION = "domain-adapter.prompts.v1"


def _normalise_domain(domain: Optional[str]) -> str:
    if not domain:
        return "business_process"
    key = str(domain).lower()
    return _DOMAIN_ALIASES.get(key, key)


class BPMNModelChecker:
    """Lightweight BPMN structural checker for formative assessments."""

    def evaluate(self, user_id: str, payload: Mapping[str, Any]) -> AssessmentResult:
        domain = _normalise_domain(payload.get("domain") or "business_process")
        item_id = str(payload.get("item_id") or payload.get("skill_id") or "bpmn.activity")
        bloom_level = str(payload.get("bloom_level") or "K3")
        model_xml = str(payload.get("model_xml") or payload.get("response") or "")
        skill_id = str(payload.get("skill_id") or f"{domain}.modeling")

        counts = Counter()
        structural_errors: List[str] = []
        try:
            root = ET.fromstring(model_xml)
            for elem in root.iter():
                tag = elem.tag.split("}")[-1]
                if tag == "startEvent":
                    counts["start"] += 1
                elif tag == "endEvent":
                    counts["end"] += 1
                elif tag.endswith("Task") or tag == "task":
                    counts["task"] += 1
                elif tag.endswith("Gateway"):
                    counts["gateway"] += 1
                elif tag == "sequenceFlow":
                    counts["flow"] += 1
        except ET.ParseError:
            structural_errors.append("invalid_xml")

        metrics = {
            "start_event": counts.get("start", 0) > 0,
            "end_event": counts.get("end", 0) > 0,
            "tasks_present": counts.get("task", 0) >= 2,
            "gateway_present": counts.get("gateway", 0) > 0,
        }
        score_components = [1.0 if flag else 0.0 for flag in metrics.values()]
        if structural_errors:
            score_components = [0.0]
        score = float(sum(score_components) / len(score_components)) if score_components else 0.0

        step_evaluations: List[AssessmentStepEvaluation] = []
        error_patterns: List[AssessmentErrorPattern] = []

        for key, flag in metrics.items():
            step_evaluations.append(
                AssessmentStepEvaluation(
                    step_id=f"bpmn.{key}",
                    subskill=f"bpmn.{key}",
                    outcome="correct" if flag else "incorrect",
                    score_delta=0.25 if flag else 0.0,
                    feedback=(
                        "Gateway fehlt im Modell. Ergänze einen Entscheidungs- oder Parallelpfad."
                        if key == "gateway_present" and not flag
                        else None
                    ),
                    diagnosis="procedural" if not flag else "none",
                )
            )
            if not flag:
                error_patterns.append(
                    AssessmentErrorPattern(
                        code=f"bpmn_missing_{key}",
                        description=f"BPMN Modell weist kein {key.replace('_', ' ')} auf.",
                        subskill=f"bpmn.{key}",
                        occurrences=1,
                    )
                )

        for err in structural_errors:
            error_patterns.append(
                AssessmentErrorPattern(
                    code=err,
                    description="BPMN XML konnte nicht geparst werden.",
                    subskill="bpmn.syntax",
                    occurrences=1,
                )
            )

        rubric = [
            RubricCriterion(id="structure", score=score),
            RubricCriterion(id="flow_connectivity", score=1.0 if counts.get("flow", 0) >= 2 else 0.5),
        ]

        diagnosis = "procedural" if score < 0.75 else "none"
        if structural_errors:
            diagnosis = "conceptual"

        return AssessmentResult(
            user_id=user_id,
            domain=domain,
            item_id=item_id,
            bloom_level=bloom_level,
            response=model_xml,
            score=round(score, 4),
            rubric_criteria=rubric,
            model_version=_ASSESSMENT_MODEL_VERSION,
            prompt_version=_PROMPT_VERSION,
            confidence=min(1.0, 0.4 + score * 0.6),
            diagnosis=diagnosis,
            source="heuristic",
            step_evaluations=step_evaluations,
            error_patterns=error_patterns,
        )


class SpeechAssessmentPipeline:
    """Approximate speech feedback using transcript coverage heuristics."""

    def evaluate(self, user_id: str, payload: Mapping[str, Any]) -> AssessmentResult:
        domain = _normalise_domain(payload.get("domain") or "language")
        item_id = str(payload.get("item_id") or payload.get("skill_id") or "language.practice")
        bloom_level = str(payload.get("bloom_level") or "K2")
        transcript = str(
            payload.get("transcript")
            or payload.get("audio_transcript")
            or payload.get("response")
            or ""
        )
        skill_id = str(payload.get("skill_id") or f"{domain}.speaking")
        target_terms = {
            str(term).lower()
            for term in payload.get("target_terms") or payload.get("word_bank") or []
        }

        if not transcript.strip():
            coverage = 0.0
        elif not target_terms:
            coverage = 1.0
        else:
            words = {token.lower() for token in re.findall(r"[\w']+", transcript)}
            coverage = len(words & target_terms) / len(target_terms)

        tone_score = float(payload.get("tone_score") or 0.5)
        pronunciation_score = float(payload.get("pronunciation_score") or coverage)
        combined = statistics.mean([coverage, pronunciation_score, tone_score]) if target_terms else coverage
        score = max(0.0, min(1.0, combined))

        step_evaluations: List[AssessmentStepEvaluation] = []
        for term in sorted(target_terms):
            outcome = "correct" if term in transcript.lower() else "incorrect"
            step_evaluations.append(
                AssessmentStepEvaluation(
                    step_id=f"speech.term.{term}",
                    subskill=f"{skill_id}.lexis",
                    outcome=outcome,
                    score_delta=1.0 / max(len(target_terms), 1) if outcome == "correct" else 0.0,
                    hint=(
                        f"Übe die Aussprache von '{term}' noch einmal mit Fokus auf Tonhöhen."
                        if outcome == "incorrect"
                        else None
                    ),
                    diagnosis="procedural" if outcome == "incorrect" else "none",
                )
            )

        error_patterns: List[AssessmentErrorPattern] = []
        if coverage < 0.5 and target_terms:
            error_patterns.append(
                AssessmentErrorPattern(
                    code="speech_low_coverage",
                    description="Weniger als die Hälfte der Zielwörter wurden verwendet.",
                    subskill=f"{skill_id}.lexis",
                    occurrences=1,
                )
            )

        rubric = [
            RubricCriterion(id="coverage", score=round(coverage, 3)),
            RubricCriterion(id="pronunciation", score=round(pronunciation_score, 3)),
        ]

        diagnosis = "procedural" if coverage < 0.5 else "none"

        return AssessmentResult(
            user_id=user_id,
            domain=domain,
            item_id=item_id,
            bloom_level=bloom_level,
            response=transcript,
            score=round(score, 4),
            rubric_criteria=rubric,
            model_version=_ASSESSMENT_MODEL_VERSION,
            prompt_version=_PROMPT_VERSION,
            confidence=min(1.0, 0.5 + score * 0.5),
            diagnosis=diagnosis,
            source="heuristic",
            step_evaluations=step_evaluations,
            error_patterns=error_patterns,
        )


class MathStepEvaluator:
    """Rule-based evaluation of step-by-step mathematics solutions."""

    _ALLOWED_NODES = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.Name,
        ast.Call,
    }
    _ALLOWED_NAMES = {"sqrt": math.sqrt, "abs": abs}

    def _safe_eval(self, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if type(node) not in self._ALLOWED_NODES:
                raise ValueError(f"Unsupported expression element: {type(node).__name__}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in self._ALLOWED_NAMES:
                    raise ValueError("Unsupported function call in expression")
        compiled = compile(tree, "<math_eval>", "eval")
        return float(eval(compiled, {"__builtins__": {}}, self._ALLOWED_NAMES))

    def evaluate(self, user_id: str, payload: Mapping[str, Any]) -> AssessmentResult:
        domain = _normalise_domain(payload.get("domain") or "mathematics")
        item_id = str(payload.get("item_id") or payload.get("skill_id") or "math.problem")
        bloom_level = str(payload.get("bloom_level") or "K3")
        skill_id = str(payload.get("skill_id") or f"{domain}.problem_solving")
        steps: Sequence[Mapping[str, Any]] = payload.get("steps") or []
        expected = payload.get("expected_result")

        correct_steps = 0
        evaluations: List[AssessmentStepEvaluation] = []
        error_patterns: Dict[str, AssessmentErrorPattern] = {}
        previous_value: Optional[float] = None

        for index, step in enumerate(steps, start=1):
            step_id = str(step.get("step_id") or index)
            expression = str(step.get("expression") or step.get("value") or "")
            subskill = str(step.get("subskill") or f"{skill_id}.step")
            try:
                value = self._safe_eval(expression)
                expected_value = step.get("expected")
                if expected_value is not None:
                    try:
                        expected_float = self._safe_eval(str(expected_value))
                    except Exception:
                        expected_float = float(expected_value)
                elif previous_value is not None:
                    expected_float = previous_value
                else:
                    expected_float = value
                is_correct = math.isclose(value, expected_float, rel_tol=1e-3, abs_tol=1e-3)
            except Exception:
                value = None
                is_correct = False

            if is_correct:
                correct_steps += 1
                outcome = "correct"
                diagnosis = "none"
            else:
                outcome = "incorrect"
                diagnosis = "procedural"
                error_patterns.setdefault(
                    subskill,
                    AssessmentErrorPattern(
                        code=f"math_step_error_{subskill}",
                        description="Zwischenschritt weist einen Rechenfehler auf.",
                        subskill=subskill,
                        occurrences=0,
                    ),
                ).occurrences += 1

            evaluations.append(
                AssessmentStepEvaluation(
                    step_id=f"math.{step_id}",
                    subskill=subskill,
                    outcome=outcome,
                    score_delta=1.0 / max(len(steps), 1) if is_correct else 0.0,
                    feedback=(
                        "Prüfe die Umformung dieses Zwischenschritts noch einmal."
                        if not is_correct
                        else None
                    ),
                    diagnosis=diagnosis,
                )
            )
            previous_value = value if value is not None else previous_value

        total_steps = max(len(steps), 1)
        score = correct_steps / total_steps
        final_correct = None
        if expected is not None and previous_value is not None:
            try:
                target_value = self._safe_eval(str(expected))
            except Exception:
                target_value = float(expected)
            final_correct = math.isclose(previous_value, target_value, rel_tol=1e-3, abs_tol=1e-3)
            if not final_correct:
                score *= 0.8
        rubric = [
            RubricCriterion(id="steps_correct", score=round(score, 3)),
        ]
        diagnosis = "procedural" if score < 0.7 else "none"

        return AssessmentResult(
            user_id=user_id,
            domain=domain,
            item_id=item_id,
            bloom_level=bloom_level,
            response=str(payload.get("response") or ""),
            score=round(score, 4),
            rubric_criteria=rubric,
            model_version=_ASSESSMENT_MODEL_VERSION,
            prompt_version=_PROMPT_VERSION,
            confidence=min(1.0, 0.45 + score * 0.55),
            diagnosis=diagnosis,
            source="heuristic",
            step_evaluations=evaluations,
            error_patterns=list(error_patterns.values()),
        )


@dataclass
class ProgressionStrategy:
    domain: str
    engine: ProgressionEngine
    rationale: str


class DomainAdaptiveOrchestrator:
    """Selects strategies, feedback tones, and modalities per domain."""

    def __init__(
        self,
        *,
        templates: Optional[Dict[str, List[CompetencyTemplate]]] = None,
        graph: Optional[KnowledgeGraph] = None,
    ) -> None:
        self.templates = templates or ALL_TEMPLATES
        self.graph = graph or build_knowledge_graph()
        self._progression: Dict[str, ProgressionStrategy] = {
            "business_process": ProgressionStrategy(
                domain="business_process",
                engine=ProgressionEngine(window_size=4, min_attempts=2, advance_threshold=0.82),
                rationale="BPMN benötigt schnelle Rückkopplung auf Modellierungsfehler.",
            ),
            "mathematics": ProgressionStrategy(
                domain="mathematics",
                engine=ProgressionEngine(window_size=6, min_attempts=3, regress_threshold=0.4),
                rationale="Mathematik profitiert von stabileren Fenstern für Schrittfolgen.",
            ),
            "language": ProgressionStrategy(
                domain="language",
                engine=ProgressionEngine(window_size=5, min_attempts=3, advance_threshold=0.75),
                rationale="Sprachlernende brauchen kontinuierliche Audio- und Dialogimpulse.",
            ),
        }
        self._pipelines: Dict[str, Any] = {
            "business_process": BPMNModelChecker(),
            "mathematics": MathStepEvaluator(),
            "language": SpeechAssessmentPipeline(),
        }

    # ------------------------------------------------------------------
    def recommend(self, competency: SkillDefinition, *, mastery: float) -> Recommendation:
        template = self._select_template(competency)
        modality = self._pick_modality(template, mastery)
        prompt = template.prompts.get("activity", "Design an adaptive activity.")
        feedback_prompt = template.prompts.get("feedback", "Provide supportive feedback.")
        metadata = {**template.metadata, **competency.metadata}
        metadata.update({"bloom_level": competency.bloom_level, "mastery": mastery})
        return Recommendation(
            competency_id=competency.to_node().identifier,
            modality=modality,
            prompt=prompt,
            feedback_prompt=feedback_prompt,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def next_competencies(self, mastered: Iterable[str]) -> List[CompetencyNode]:
        return self.graph.ready_nodes(mastered)

    # ------------------------------------------------------------------
    def progression_engine_for(self, domain: str) -> ProgressionEngine:
        key = _normalise_domain(domain)
        if key not in self._progression:
            self._progression[key] = ProgressionStrategy(
                domain=key,
                engine=ProgressionEngine(),
                rationale="Fallback progression engine.",
            )
        return self._progression[key].engine

    # ------------------------------------------------------------------
    def select_feedback_tone(self, domain: str, signal: Mapping[str, Any]) -> str:
        key = _normalise_domain(domain)
        profile = {
            "business_process": ("coaching", "analytical", "celebratory"),
            "mathematics": ("socratic", "supportive", "celebratory"),
            "language": ("supportive", "dialogic", "celebratory"),
        }.get(key, ("supportive", "coaching", "celebratory"))

        confidence = float(signal.get("confidence") or 0.5)
        recent = str(signal.get("recent_outcome") or "unknown").lower()
        frustration = float(signal.get("frustration") or 0.0)

        if recent in {"failed", "incorrect"} or confidence < 0.4 or frustration >= 0.5:
            return profile[0]
        if confidence > 0.85 and recent in {"success", "correct", "mastered"}:
            return profile[-1]
        return profile[min(1, len(profile) - 1)]

    # ------------------------------------------------------------------
    def prioritise_modalities(self, domain: str, *, skill_id: Optional[str] = None) -> List[Dict[str, Any]]:
        key = _normalise_domain(domain)
        preference = list(_MEDIA_PRIORITIES.get(key, ("text",)))
        counts: Counter[str] = Counter()

        nodes = self._matching_nodes(key, skill_id=skill_id)
        for node in nodes:
            for resource in self.graph.iter_node_resources(node.identifier):
                cluster = resource.media_cluster or resource.modality
                counts[cluster] += 1

        if not counts:
            for cluster in preference:
                counts.setdefault(cluster, 0)

        def sort_key(item: Tuple[str, int]) -> Tuple[int, float, str]:
            cluster, count = item
            try:
                order_index = preference.index(cluster)
            except ValueError:
                order_index = len(preference)
            return (order_index, -count, cluster)

        ordered = sorted(counts.items(), key=sort_key)
        return [{"channel": cluster, "count": count} for cluster, count in ordered]

    # ------------------------------------------------------------------
    def evaluate_assessment(
        self,
        domain: str,
        user_id: str,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        key = _normalise_domain(domain)
        pipeline = self._pipelines.get(key)
        if not pipeline:
            raise KeyError(f"No assessment pipeline registered for domain '{domain}'")
        result = pipeline.evaluate(user_id, payload)
        assessment_id = db.save_assessment_result(result)
        return {"assessment_id": assessment_id, "result": result}

    # ------------------------------------------------------------------
    def _select_template(self, competency: SkillDefinition) -> CompetencyTemplate:
        options = self.templates.get(competency.domain, [])
        if not options:
            options = self.templates.get(_normalise_domain(competency.domain), [])
        if not options:
            raise KeyError(f"No templates defined for domain {competency.domain}")
        for template in options:
            lower, upper = template.bloom_range
            if lower <= competency.bloom_level <= upper:
                return template
        return options[-1]

    # ------------------------------------------------------------------
    def _pick_modality(self, template: CompetencyTemplate, mastery: float) -> str:
        if mastery < 0.4:
            return template.modalities[0]
        if mastery < 0.7 and len(template.modalities) > 1:
            return template.modalities[1]
        return template.modalities[-1]

    # ------------------------------------------------------------------
    def _matching_nodes(
        self,
        domain: str,
        *,
        skill_id: Optional[str] = None,
    ) -> List[CompetencyNode]:
        matches: Dict[str, CompetencyNode] = {}
        if skill_id:
            direct = self.graph.get_node(skill_id)
            if direct:
                matches[direct.identifier] = direct
            else:
                candidates = self.graph.find_nodes(skill_ids=[skill_id])
                for node in candidates:
                    matches[node.identifier] = node
                if ":" in skill_id:
                    parts = skill_id.split(":")
                    if len(parts) >= 2:
                        simple_id = parts[1]
                        for node in self.graph.find_nodes(skill_ids=[simple_id]):
                            matches[node.identifier] = node
        if not matches:
            for node in self.graph.nodes():
                if _normalise_domain(node.domain) == domain:
                    matches[node.identifier] = node
        return list(matches.values())


__all__ = ["DomainAdaptiveOrchestrator", "BPMNModelChecker", "SpeechAssessmentPipeline", "MathStepEvaluator"]
