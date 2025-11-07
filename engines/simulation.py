"""Simulation utilities for adaptive learning policy evaluation."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from engines.competency import (
    DomainAdaptiveOrchestrator,
    Recommendation,
    SKILL_REGISTRY,
    SkillDefinition,
)
from knowledge_graph import CompetencyNode


@dataclass
class Persona:
    """Represents a simulated learner profile."""

    name: str
    accuracy_bias: float
    confidence_bias: float
    latency_bias: float


@dataclass
class EpisodeResult:
    """Outcome of a single learner interaction with a competency."""

    persona: str
    competency_id: str
    mastery: float
    modality: str
    success: bool
    steps: int


@dataclass
class SimulationMetrics:
    """Aggregated statistics for a persona across simulation episodes."""

    persona: str
    episodes: int
    completion_rate: float
    mean_mastery: float
    mean_steps_to_mastery: float
    modality_distribution: Dict[str, int]


AttemptModel = Callable[[random.Random, Persona, float, Recommendation], Tuple[bool, float]]


class LearningSimulation:
    """Evaluate learning pathways with Monte Carlo and trace replay techniques."""

    def __init__(
        self,
        *,
        orchestrator: DomainAdaptiveOrchestrator,
        personas: Sequence[Persona] | None = None,
        random_seed: int | None = None,
        attempt_model: AttemptModel | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.personas: List[Persona] = list(personas) if personas is not None else [
            Persona(name="Novice", accuracy_bias=0.55, confidence_bias=0.4, latency_bias=1.2),
            Persona(name="FastAdvancer", accuracy_bias=0.82, confidence_bias=0.75, latency_bias=0.8),
            Persona(name="SlowConfidence", accuracy_bias=0.65, confidence_bias=0.3, latency_bias=1.4),
        ]
        self.rng = random.Random(random_seed)
        self.attempt_model: AttemptModel = attempt_model or self._default_attempt_model

    # ------------------------------------------------------------------
    def run(self, *, steps: int = 10, iterations: int = 1) -> List[EpisodeResult]:
        """Backward-compatible alias for :meth:`run_monte_carlo`."""

        return self.run_monte_carlo(steps=steps, iterations=iterations)

    # ------------------------------------------------------------------
    def run_monte_carlo(self, *, steps: int = 10, iterations: int = 30) -> List[EpisodeResult]:
        """Run Monte Carlo simulations across personas."""

        results: List[EpisodeResult] = []
        for _ in range(iterations):
            for persona in self.personas:
                results.extend(self._simulate_persona(persona, steps=steps))
        return results

    # ------------------------------------------------------------------
    def run_trace_replay(
        self,
        *,
        persona_name: str,
        competency_sequence: Sequence[str],
        initial_mastery: float = 0.3,
    ) -> List[EpisodeResult]:
        """Replay a deterministic competency sequence (e.g., mined from logs)."""

        persona = self._persona_by_name(persona_name)
        mastered: List[str] = []
        mastery_state: Dict[str, float] = defaultdict(lambda: initial_mastery)
        results: List[EpisodeResult] = []
        for competency_id in competency_sequence:
            skill = self._skill_from_node(competency_id)
            mastery = mastery_state[competency_id]
            recommendation = self.orchestrator.recommend(skill, mastery=mastery)
            success, delta = self.attempt_model(self.rng, persona, mastery, recommendation)
            mastery_state[competency_id] = max(0.0, min(1.0, mastery + delta))
            if success and mastery_state[competency_id] >= 0.85 and competency_id not in mastered:
                mastered.append(competency_id)
            results.append(
                EpisodeResult(
                    persona=persona.name,
                    competency_id=competency_id,
                    mastery=mastery_state[competency_id],
                    modality=recommendation.modality,
                    success=success,
                    steps=len(mastered),
                )
            )
        return results

    # ------------------------------------------------------------------
    def summarise(self, episodes: Iterable[EpisodeResult], *, mastery_threshold: float = 0.85) -> List[SimulationMetrics]:
        """Aggregate metrics for reporting and experiment dashboards."""

        grouped: Dict[str, List[EpisodeResult]] = defaultdict(list)
        for episode in episodes:
            grouped[episode.persona].append(episode)

        summaries: List[SimulationMetrics] = []
        for persona, persona_episodes in grouped.items():
            completions = [
                ep.steps
                for ep in persona_episodes
                if ep.success and ep.mastery >= mastery_threshold
            ]
            modality_distribution = Counter(ep.modality for ep in persona_episodes)
            summaries.append(
                SimulationMetrics(
                    persona=persona,
                    episodes=len(persona_episodes),
                    completion_rate=len(completions) / len(persona_episodes) if persona_episodes else 0.0,
                    mean_mastery=mean(ep.mastery for ep in persona_episodes) if persona_episodes else 0.0,
                    mean_steps_to_mastery=mean(completions) if completions else 0.0,
                    modality_distribution=dict(modality_distribution),
                )
            )
        return summaries

    # ------------------------------------------------------------------
    def _simulate_persona(self, persona: Persona, *, steps: int) -> List[EpisodeResult]:
        mastered: List[str] = []
        mastery_state: Dict[str, float] = {}
        results: List[EpisodeResult] = []
        for _ in range(steps):
            next_nodes = self.orchestrator.next_competencies(mastered)
            if not next_nodes:
                next_nodes = self._fallback_nodes()
            if not next_nodes:
                break
            node = self.rng.choice(next_nodes)
            competency_id = node.identifier
            skill = self._skill_from_node(competency_id)
            mastery = mastery_state.get(competency_id, 0.3)
            recommendation = self.orchestrator.recommend(skill, mastery=mastery)
            success, delta = self.attempt_model(self.rng, persona, mastery, recommendation)
            mastery_state[competency_id] = max(0.0, min(1.0, mastery + delta))
            if success and mastery_state[competency_id] >= 0.85 and competency_id not in mastered:
                mastered.append(competency_id)
            results.append(
                EpisodeResult(
                    persona=persona.name,
                    competency_id=competency_id,
                    mastery=mastery_state[competency_id],
                    modality=recommendation.modality,
                    success=success,
                    steps=len(mastered),
                )
            )
        return results

    # ------------------------------------------------------------------
    def _skill_from_node(self, node_id: str) -> SkillDefinition:
        domain, skill_id, *_ = node_id.split(":")
        for skill in SKILL_REGISTRY.get(domain, []):
            if skill.skill_id == skill_id:
                return skill
        raise KeyError(f"Skill {node_id} not found in registry")

    # ------------------------------------------------------------------
    def _persona_by_name(self, name: str) -> Persona:
        for persona in self.personas:
            if persona.name == name:
                return persona
        raise ValueError(f"Persona {name} not defined")

    # ------------------------------------------------------------------
    def _fallback_nodes(self) -> List[CompetencyNode]:
        nodes: List[CompetencyNode] = []
        for skills in SKILL_REGISTRY.values():
            nodes.extend(skill.to_node() for skill in skills)
        return nodes

    # ------------------------------------------------------------------
    @staticmethod
    def _default_attempt_model(
        rng: random.Random,
        persona: Persona,
        mastery: float,
        recommendation: Recommendation,
    ) -> Tuple[bool, float]:
        """Heuristic attempt model balancing mastery, persona bias, and modality."""

        modality_factor = 0.05 if recommendation.modality in {"video", "flashcards"} else 0.0
        effective_accuracy = mastery * 0.5 + persona.accuracy_bias * 0.4 + 0.1 + modality_factor
        effective_accuracy = max(0.05, min(0.95, effective_accuracy))
        success = rng.random() < effective_accuracy
        adjustment = 0.1 if success else -0.06
        adjustment *= 0.8 + persona.confidence_bias
        return success, adjustment


__all__ = [
    "Persona",
    "EpisodeResult",
    "SimulationMetrics",
    "LearningSimulation",
]

