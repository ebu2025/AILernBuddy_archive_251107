import pytest

from knowledge_graph import KnowledgeGraph, register_language_modules


def test_register_language_modules_loads_yaml(tmp_path):
    pytest.importorskip("yaml")
    yaml_text = "domain: language.example\nskills:\n  - id: example-skill\n    name: Example Skill\n    module: example_module\n    bloom_level: K2\n    resources:\n      - resource_id: res-audio\n        title: Audio Clip\n        uri: audio/example.mp3\n        modality: audio\n"
    matrix_path = tmp_path / "language.example.yaml"
    matrix_path.write_text(yaml_text, encoding="utf-8")

    graph = KnowledgeGraph()
    summary = register_language_modules(graph, matrix_paths=[matrix_path])
    assert summary["matrices_loaded"] == 1
    nodes = graph.find_nodes(domain="language.example")
    assert nodes
    node_id = nodes[0].identifier
    resources = graph.get_resources(node_id)
    assert resources and resources[0].media_cluster == "audio"
    clusters = graph.resource_clusters(node_id)
    assert clusters.get("audio") == 1
