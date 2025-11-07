from knowledge_graph import (
    CompetencyEdge,
    CompetencyNode,
    ContentResource,
    KnowledgeGraph,
    register_math_modules,
)


def build_sample_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    nodes = [
        CompetencyNode("data", "ingest", "Ingest Data", "K2"),
        CompetencyNode("data", "model", "Model Data", "K3"),
        CompetencyNode("data", "deploy", "Deploy Models", "K4"),
    ]
    edges = [
        CompetencyEdge(nodes[0].identifier, nodes[1].identifier),
        CompetencyEdge(nodes[1].identifier, nodes[2].identifier),
    ]
    graph.add_module("data_pipeline", nodes, edges)
    graph.link_resource(
        nodes[2].identifier,
        ContentResource(
            resource_id="vid-1",
            title="Deploying Models Walkthrough",
            uri="https://example.com/deploy",
            modality="video",
            metadata={"duration": 540},
        ),
    )
    return graph


def test_module_round_trip_and_resources():
    graph = build_sample_graph()
    assert set(graph.module_names()) == {"data_pipeline"}

    serialized = graph.to_dict()
    restored = KnowledgeGraph.from_dict(serialized)

    assert restored.module_nodes("data_pipeline") == {
        node.identifier for node in graph.find_nodes(domain="data")
    }
    deploy_node = restored.find_nodes(domain="data", skill_ids=["deploy"])[0]
    resources = restored.get_resources(deploy_node.identifier)
    assert resources and resources[0].uri == "https://example.com/deploy"


def test_ancestor_lookup_and_find_nodes():
    graph = build_sample_graph()
    deploy_node = graph.find_nodes(domain="data", skill_ids=["deploy"])[0]
    ancestors = graph.ancestors(deploy_node.identifier)
    assert any("ingest" in ancestor for ancestor in ancestors)
    assert any(node.skill_id == "model" for node in graph.find_nodes(domain="data", bloom_levels=["K3"]))


def test_register_math_modules_enriches_graph_with_misconceptions():
    graph = KnowledgeGraph()
    summary = register_math_modules(graph)

    assert summary["modules_registered"] >= 1
    assert any(name.startswith("math_") for name in graph.module_names())

    math_nodes = graph.find_nodes(domain="math")
    assert any(node.metadata.get("type") == "misconception" for node in math_nodes)
    assert any(node.metadata.get("type") == "application" for node in math_nodes)

    algebra_nodes = [node for node in math_nodes if node.skill_id == "algebra.linear_equations"]
    assert algebra_nodes, "Expected algebra.linear_equations node to be registered"
    resources = graph.get_resources(algebra_nodes[0].identifier)
    assert any("Desmos" in resource.title for resource in resources)
