"""Knowledge graph and process mining driven learning path planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from knowledge_graph import (
    CompetencyEdge,
    CompetencyNode,
    ContentResource,
    KnowledgeGraph,
)
from process_models.process_mining import (
    Event,
    generate_process_diagnostics,
    parse_event_log,
)


@dataclass
class LearnerProfile:
    """Minimal representation of a learner's current mastery state."""

    user_id: str
    mastered_nodes: Set[str] = field(default_factory=set)
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningGoal:
    """Target skills or competencies to reach."""

    domain: str
    skill_ids: Sequence[str]
    target_bloom: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPathPlan:
    """Structured output of a path recommendation."""

    ordered_nodes: List[CompetencyNode]
    resources: Dict[str, List[ContentResource]]
    diagnostics: Dict[str, Any]
    insights: List[str]
    skill_success: Dict[str, float]
    preference_matches: Dict[str, List[str]] = field(default_factory=dict)
    preference_highlights: List[str] = field(default_factory=list)


class KnowledgeProcessOrchestrator:
    """Combine knowledge graph structure with process mining signals."""

    def __init__(self, graph: Optional[KnowledgeGraph] = None) -> None:
        self.graph = graph or KnowledgeGraph()

    # ------------------------------------------------------------------
    def register_module(
        self,
        name: str,
        nodes: Sequence[CompetencyNode],
        edges: Sequence[CompetencyEdge],
    ) -> None:
        self.graph.add_module(name, nodes, edges)

    # ------------------------------------------------------------------
    def add_resource(self, node_id: str, resource: ContentResource) -> None:
        self.graph.link_resource(node_id, resource)

    # ------------------------------------------------------------------
    def recommend_path(
        self,
        profile: LearnerProfile,
        goals: Sequence[LearningGoal],
        event_log: Iterable[Dict[str, Any]],
        *,
        end_activity: Optional[str] = None,
    ) -> LearningPathPlan:
        events = parse_event_log(event_log or [])
        diagnostics = generate_process_diagnostics(events, end_activity=end_activity)
        success_rates = self._skill_success_rates(events)
        goal_skill_ids = {skill for goal in goals for skill in goal.skill_ids}
        preference_hits: Dict[str, List[str]] = {}

        relevant_nodes = self._collect_goal_nodes(goals)
        if not relevant_nodes:
            # fall back to unlocked nodes if goals cannot be mapped directly
            relevant_nodes = {node.identifier for node in self.graph.ready_nodes(profile.mastered_nodes)}

        # Optimise edge weights based on observed success rates
        self._rebalance_edge_weights(relevant_nodes, success_rates)

        ordered_ids = self._sequence_nodes(
            relevant_nodes,
            profile,
            success_rates,
            goal_skill_ids,
            preference_hits,
        )
        ordered_nodes = [self.graph.get_node(node_id) for node_id in ordered_ids if self.graph.get_node(node_id)]
        resources = {node_id: self.graph.get_resources(node_id) for node_id in ordered_ids}
        preference_highlights = self._build_preference_highlights(
            ordered_nodes,
            preference_hits,
            profile.preferences,
        )
        insights = self._build_insights(ordered_nodes, success_rates, diagnostics)
        if preference_highlights:
            insights.extend(preference_highlights)

        return LearningPathPlan(
            ordered_nodes=ordered_nodes,
            resources=resources,
            diagnostics=diagnostics,
            insights=insights,
            skill_success=success_rates,
            preference_matches=preference_hits,
            preference_highlights=preference_highlights,
        )

    # ------------------------------------------------------------------
    def _collect_goal_nodes(self, goals: Sequence[LearningGoal]) -> Set[str]:
        relevant_nodes: Set[str] = set()
        for goal in goals:
            bloom_filter = [goal.target_bloom] if goal.target_bloom else None
            candidates = self.graph.find_nodes(
                domain=goal.domain,
                skill_ids=goal.skill_ids,
                bloom_levels=bloom_filter,
            )
            if not candidates:
                # allow broader match if specific bloom level missing
                candidates = self.graph.find_nodes(domain=goal.domain, skill_ids=goal.skill_ids)
            for node in candidates:
                relevant_nodes.add(node.identifier)
                relevant_nodes.update(self.graph.ancestors(node.identifier))
        return relevant_nodes

    # ------------------------------------------------------------------
    def _rebalance_edge_weights(
        self,
        node_ids: Set[str],
        success_rates: Dict[str, float],
    ) -> None:
        for node_id in node_ids:
            node = self.graph.get_node(node_id)
            if not node:
                continue
            success = success_rates.get(node.skill_id)
            if success is None:
                continue
            for edge in self.graph.dependencies_of(node_id):
                if edge.relation != "prerequisite":
                    continue
                try:
                    weight = max(0.1, 1.5 - success)
                    self.graph.update_edge_weight(edge.source, edge.target, weight=weight)
                except KeyError:
                    continue

    # ------------------------------------------------------------------
    def _sequence_nodes(
        self,
        candidate_ids: Set[str],
        profile: LearnerProfile,
        success_rates: Dict[str, float],
        goal_skills: Set[str],
        preference_hits: Dict[str, List[str]],
    ) -> List[str]:
        mastered = set(profile.mastered_nodes)
        ordered: List[str] = []
        remaining = set(candidate_ids) - mastered
        while remaining:
            ready: List[str] = []
            for node_id in list(remaining):
                prerequisites = {
                    edge.source
                    for edge in self.graph.dependencies_of(node_id)
                    if edge.relation == "prerequisite"
                }
                if prerequisites.issubset(mastered):
                    ready.append(node_id)
            if not ready:
                break
            ready.sort(
                key=lambda nid: self._node_priority(
                    self.graph.get_node(nid),
                    success_rates,
                    profile,
                    goal_skills,
                    preference_hits,
                ),
                reverse=True,
            )
            selected = ready[0]
            ordered.append(selected)
            mastered.add(selected)
            remaining.remove(selected)
        return ordered

    # ------------------------------------------------------------------
    def _node_priority(
        self,
        node: Optional[CompetencyNode],
        success_rates: Dict[str, float],
        profile: LearnerProfile,
        goal_skills: Set[str],
        preference_hits: Dict[str, List[str]],
    ) -> float:
        if not node:
            return 0.0
        priority = 1.0
        matches: List[str] = []
        success = success_rates.get(node.skill_id)
        if success is not None:
            priority += (1.0 - success)
        if node.skill_id in goal_skills:
            priority += 0.5
        preferred_modalities = self._preferred_modalities(profile.preferences)
        if preferred_modalities:
            resources = self.graph.get_resources(node.identifier)
            matching_modalities = {
                resource.modality
                for resource in resources
                if resource.modality in preferred_modalities
            }
            if matching_modalities:
                priority += 0.1
                matches.append("modalities: " + ", ".join(sorted(matching_modalities)))
        focus_bloom = profile.preferences.get("focus_bloom")
        if focus_bloom and node.bloom_level == focus_bloom:
            priority += 0.2
            matches.append(f"focus_bloom {focus_bloom}")
        if matches:
            preference_hits[node.identifier] = matches
        else:
            preference_hits.setdefault(node.identifier, [])
        return priority

    # ------------------------------------------------------------------
    @staticmethod
    def _preferred_modalities(preferences: Dict[str, Any]) -> Set[str]:
        modalities: Set[str] = set()
        for key in ("preferred_modalities", "modalities"):
            value = preferences.get(key)
            if isinstance(value, (list, tuple, set)):
                for entry in value:
                    text = str(entry).strip().lower()
                    if text:
                        modalities.add(text)
        return modalities

    # ------------------------------------------------------------------
    @staticmethod
    def _skill_success_rates(events: Sequence[Event]) -> Dict[str, float]:
        attempts: Dict[str, int] = {}
        successes: Dict[str, int] = {}
        for event in events:
            skill_id = (
                event.metadata.get("skill_id")
                or event.metadata.get("competency_id")
                or event.metadata.get("skill")
            )
            if not skill_id:
                continue
            attempts[skill_id] = attempts.get(skill_id, 0) + 1
            outcome = str(event.metadata.get("outcome") or event.status).lower()
            if outcome in {"complete", "completed", "passed", "success"}:
                successes[skill_id] = successes.get(skill_id, 0) + 1
        rates: Dict[str, float] = {}
        for skill_id, count in attempts.items():
            rates[skill_id] = round(successes.get(skill_id, 0) / count, 3)
        return rates

    # ------------------------------------------------------------------
    @staticmethod
    def _build_insights(
        nodes: Sequence[CompetencyNode],
        success_rates: Dict[str, float],
        diagnostics: Dict[str, Any],
    ) -> List[str]:
        insights: List[str] = []
        dropout = diagnostics.get("dropouts", {})
        dropout_rate = dropout.get("rate")
        if dropout_rate:
            insights.append(
                f"Dropout rate across analysed journeys is {dropout_rate:.0%}; include motivational checkpoints."
            )
        for node in nodes:
            success = success_rates.get(node.skill_id)
            if success is None:
                continue
            if success < 0.6:
                insights.append(
                    f"{node.label} ({node.skill_id}) shows {success:.0%} success – add remediation resources."
                )
            elif success > 0.85:
                insights.append(
                    f"{node.label} ({node.skill_id}) is a strength with {success:.0%} success – accelerate progression."
                )
        return insights

    # ------------------------------------------------------------------
    def _build_preference_highlights(
        self,
        nodes: Sequence[CompetencyNode],
        preference_hits: Dict[str, List[str]],
        preferences: Dict[str, Any],
    ) -> List[str]:
        highlights: List[str] = []
        preferred_modalities = self._preferred_modalities(preferences)
        for node in nodes:
            if not node:
                continue
            hits = preference_hits.get(node.identifier) or []
            if not hits:
                continue
            joined = "; ".join(hits)
            highlights.append(f"{node.label} prioritised due to {joined}.")
        if not highlights and preferred_modalities:
            friendly = ", ".join(sorted(preferred_modalities))
            highlights.append(
                f"No current activities matched the preferred modalities ({friendly}); showing closest options instead."
            )
        return highlights


__all__ = [
    "LearnerProfile",
    "LearningGoal",
    "LearningPathPlan",
    "KnowledgeProcessOrchestrator",
]
