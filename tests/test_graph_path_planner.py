from engines.graph_path_planner import (
    KnowledgeProcessOrchestrator,
    LearnerProfile,
    LearningGoal,
)
from knowledge_graph import CompetencyEdge, CompetencyNode, ContentResource, KnowledgeGraph


def build_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    intro = CompetencyNode("ml", "intro", "Intro to Pipelines", "K2")
    model = CompetencyNode("ml", "model", "Build Models", "K3")
    deploy = CompetencyNode("ml", "deploy", "Deploy Models", "K5")
    graph.add_module(
        "ml_pipeline",
        [intro, model, deploy],
        [
            CompetencyEdge(intro.identifier, model.identifier),
            CompetencyEdge(model.identifier, deploy.identifier),
        ],
    )
    graph.link_resource(
        deploy.identifier,
        ContentResource(
            resource_id="res-video",
            title="Deploy with CI/CD",
            uri="https://example.com/cicd",
            modality="video",
            metadata={"length_minutes": 12},
        ),
    )
    return graph


def build_event(timestamp: str, case: str, activity: str, skill: str, outcome: str):
    return {
        "case_id": case,
        "activity": activity,
        "timestamp": timestamp,
        "metadata": {"skill_id": skill, "outcome": outcome},
    }


def test_recommend_path_prioritises_challenging_skills():
    graph = build_graph()
    orchestrator = KnowledgeProcessOrchestrator(graph)

    profile = LearnerProfile(
        user_id="learner-1",
        mastered_nodes={graph.find_nodes(domain="ml", skill_ids=["intro"])[0].identifier},
        preferences={"preferred_modalities": ["video"]},
    )
    goals = [LearningGoal(domain="ml", skill_ids=["deploy"], target_bloom="K5")]

    event_log = [
        build_event("2024-01-01T09:00:00", "case-1", "Intro", "intro", "success"),
        build_event("2024-01-01T11:00:00", "case-1", "Model", "model", "success"),
        build_event("2024-01-01T13:30:00", "case-1", "Deploy", "deploy", "failure"),
        build_event("2024-01-02T09:15:00", "case-2", "Intro", "intro", "success"),
        build_event("2024-01-02T11:45:00", "case-2", "Model", "model", "success"),
    ]

    plan = orchestrator.recommend_path(profile, goals, event_log, end_activity="Deploy")

    skills = [node.skill_id for node in plan.ordered_nodes]
    assert skills == ["model", "deploy"]
    deploy_node = graph.find_nodes(domain="ml", skill_ids=["deploy"])[0]
    assert plan.resources[deploy_node.identifier][0].modality == "video"
    assert deploy_node.identifier in plan.preference_matches
    assert plan.preference_matches[deploy_node.identifier]
    assert any("modalities" in hit for hit in plan.preference_matches[deploy_node.identifier])
    assert plan.preference_highlights
    assert plan.diagnostics["dropouts"]["rate"] > 0.0
    assert plan.skill_success["deploy"] < plan.skill_success["model"]
    assert any("Deploy Models" in insight for insight in plan.insights)
