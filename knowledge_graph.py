"""Lightweight knowledge graph for competency dependencies and resources."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
from collections import Counter, defaultdict
import json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _domain_key(domain: str) -> str:
    base = domain.split(".", 1)[0]
    if base == "language":
        return "language"
    if base in {"business", "bpmn"}:
        return "business_process"
    if base in {"math", "mathematics"}:
        return "mathematics"
    return base


def _load_matrix_payload(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    text = Path(path).read_text(encoding="utf-8")
    if suffix in {".json", ".jsonc"}:
        return json.loads(text)
    if suffix in {".yml", ".yaml"}:
        if yaml is None:  # pragma: no cover - dependency guard
            raise RuntimeError("PyYAML is required to load YAML skill matrices")
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported skill matrix format: {path}")


def _determine_media_cluster(domain: str, modality: str, metadata: Mapping[str, Any]) -> str:
    key = _domain_key(domain)
    modality_key = (modality or "").lower()
    cluster = None
    if key == "language":
        if any(token in modality_key for token in {"audio", "listening", "dialogue", "speaking"}):
            cluster = "audio"
        elif "transcript" in modality_key or "reading" in modality_key:
            cluster = "transcript"
    elif key == "business_process":
        if any(token in modality_key for token in {"diagram", "model", "bpmn", "editor"}):
            cluster = "diagram_tools"
        elif "simulation" in modality_key or "mining" in modality_key:
            cluster = "simulation"
    elif key == "mathematics":
        if any(token in modality_key for token in {"plot", "graph", "visual"}):
            cluster = "plotter"
        elif "notebook" in modality_key or "step" in modality_key:
            cluster = "notebook"
    if cluster:
        return cluster

    keywords = {
        "audio_prompts": "audio",
        "diagram_tools": "diagram_tools",
        "plotter": "plotter",
        "step_hints": "step_hints",
    }
    for key_name, value in keywords.items():
        if key_name in metadata:
            return value
    return modality_key or "other"


@dataclass(frozen=True)
class CompetencyNode:
    """Represents a competency in the multi-dimensional matrix."""

    domain: str
    skill_id: str
    label: str
    bloom_level: str
    proficiency_level: Optional[str] = None
    hsk_level: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def identifier(self) -> str:
        prof = self.proficiency_level or "-"
        hsk = f"HSK{self.hsk_level}" if self.hsk_level is not None else "-"
        return f"{self.domain}:{self.skill_id}:{self.bloom_level}:{prof}:{hsk}"


@dataclass
class CompetencyEdge:
    """Dependency between two competencies."""

    source: str
    target: str
    relation: str = "prerequisite"
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class ContentResource:
    """Learning content associated with a competency node."""

    resource_id: str
    title: str
    uri: str
    modality: str
    media_cluster: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "title": self.title,
            "uri": self.uri,
            "modality": self.modality,
            "media_cluster": self.media_cluster,
            "metadata": self.metadata,
        }


class KnowledgeGraph:
    """Graph storing competency dependencies and metadata."""

    def __init__(self) -> None:
        self._nodes: Dict[str, CompetencyNode] = {}
        self._edges: Dict[str, List[CompetencyEdge]] = {}
        self._reverse_edges: Dict[str, List[CompetencyEdge]] = {}
        self._modules: Dict[str, Set[str]] = {}
        self._module_edges: Dict[str, Set[Tuple[str, str, str]]] = {}
        self._resources: Dict[str, List[ContentResource]] = {}
        self._resource_clusters: Dict[str, Counter[str]] = defaultdict(Counter)
        self._resource_index: Dict[str, Tuple[str, ContentResource]] = {}

    # ------------------------------------------------------------------
    def add_node(self, node: CompetencyNode) -> None:
        self._nodes[node.identifier] = node
        self._edges.setdefault(node.identifier, [])
        self._reverse_edges.setdefault(node.identifier, [])
        self._resources.setdefault(node.identifier, [])
        self._resource_clusters.setdefault(node.identifier, Counter())

    # ------------------------------------------------------------------
    def add_edge(self, edge: CompetencyEdge) -> None:
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise KeyError("Both source and target must exist before creating an edge")
        self._edges.setdefault(edge.source, []).append(edge)
        self._reverse_edges.setdefault(edge.target, []).append(edge)

    # ------------------------------------------------------------------
    def update_edge_weight(self, source: str, target: str, *, weight: float) -> None:
        for edge in self._edges.get(source, []):
            if edge.target == target:
                edge.weight = round(weight, 4)
                return
        raise KeyError(f"Edge {source}->{target} not found")

    # ------------------------------------------------------------------
    def add_module(
        self,
        name: str,
        nodes: Sequence[CompetencyNode],
        edges: Sequence[CompetencyEdge],
    ) -> None:
        """Register a cohesive module of nodes and edges."""

        node_ids: Set[str] = set()
        for node in nodes:
            self.add_node(node)
            node_ids.add(node.identifier)
        self._modules.setdefault(name, set()).update(node_ids)

        edge_refs: Set[Tuple[str, str, str]] = set()
        for edge in edges:
            self.add_edge(edge)
            edge_refs.add((edge.source, edge.target, edge.relation))
        if edge_refs:
            self._module_edges.setdefault(name, set()).update(edge_refs)

    # ------------------------------------------------------------------
    def module_names(self) -> List[str]:
        return sorted(self._modules)

    # ------------------------------------------------------------------
    def module_nodes(self, name: str) -> Set[str]:
        return set(self._modules.get(name, set()))

    # ------------------------------------------------------------------
    def find_nodes(
        self,
        *,
        domain: Optional[str] = None,
        skill_ids: Optional[Iterable[str]] = None,
        bloom_levels: Optional[Iterable[str]] = None,
    ) -> List[CompetencyNode]:
        """Return nodes matching optional filters."""

        skill_set = set(skill_ids or []) or None
        bloom_set = set(bloom_levels or []) or None
        matched: List[CompetencyNode] = []
        for node in self._nodes.values():
            if domain and node.domain != domain:
                continue
            if skill_set and node.skill_id not in skill_set:
                continue
            if bloom_set and node.bloom_level not in bloom_set:
                continue
            matched.append(node)
        return matched

    # ------------------------------------------------------------------
    def get_node(self, node_id: str) -> Optional[CompetencyNode]:
        return self._nodes.get(node_id)

    # ------------------------------------------------------------------
    def nodes(self) -> List[CompetencyNode]:
        return list(self._nodes.values())

    # ------------------------------------------------------------------
    def link_resource(self, node_id: str, resource: ContentResource) -> None:
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} is not registered in the graph")
        self._resources.setdefault(node_id, [])
        existing_ids = {res.resource_id for res in self._resources[node_id]}
        if resource.resource_id in existing_ids:
            return
        cluster = resource.media_cluster or _determine_media_cluster(
            self._nodes[node_id].domain, resource.modality, resource.metadata
        )
        enriched = ContentResource(
            resource_id=resource.resource_id,
            title=resource.title,
            uri=resource.uri,
            modality=resource.modality,
            media_cluster=cluster,
            metadata=dict(resource.metadata),
        )
        self._resources[node_id].append(enriched)
        self._resource_clusters[node_id][cluster] += 1
        self._resource_index[enriched.resource_id] = (node_id, enriched)

    # ------------------------------------------------------------------
    def get_resources(self, node_id: str) -> List[ContentResource]:
        return list(self._resources.get(node_id, []))

    # ------------------------------------------------------------------
    def iter_node_resources(self, node_id: str) -> Iterator[ContentResource]:
        for resource in self._resources.get(node_id, []):
            yield resource

    # ------------------------------------------------------------------
    def resource_clusters(self, node_id: str) -> Dict[str, int]:
        return dict(self._resource_clusters.get(node_id, Counter()))

    # ------------------------------------------------------------------
    def lookup_resource(self, resource_id: str) -> Optional[Tuple[str, ContentResource]]:
        return self._resource_index.get(resource_id)

    # ------------------------------------------------------------------
    def iter_resources(
        self, *, domain: Optional[str] = None
    ) -> Iterator[Tuple[str, ContentResource]]:
        for node_id, resources in self._resources.items():
            node = self._nodes.get(node_id)
            if domain and node and node.domain != domain:
                continue
            for resource in resources:
                yield node_id, resource

    # ------------------------------------------------------------------
    def update_node_metadata(self, node_id: str, **metadata: Any) -> None:
        node = self._nodes.get(node_id)
        if not node:
            raise KeyError(f"Node {node_id} does not exist")
        updated_metadata = {**node.metadata, **metadata}
        self._nodes[node_id] = CompetencyNode(
            domain=node.domain,
            skill_id=node.skill_id,
            label=node.label,
            bloom_level=node.bloom_level,
            proficiency_level=node.proficiency_level,
            hsk_level=node.hsk_level,
            metadata=updated_metadata,
        )

    # ------------------------------------------------------------------
    def ancestors(
        self,
        node_id: str,
        *,
        relation: str = "prerequisite",
    ) -> Set[str]:
        visited: Set[str] = set()
        stack: List[str] = [node_id]
        while stack:
            current = stack.pop()
            for edge in self.dependencies_of(current):
                if relation and edge.relation != relation:
                    continue
                source = edge.source
                if source not in visited:
                    visited.add(source)
                    stack.append(source)
        return visited

    # ------------------------------------------------------------------
    def dependencies_of(self, node_id: str) -> List[CompetencyEdge]:
        return list(self._reverse_edges.get(node_id, []))

    # ------------------------------------------------------------------
    def dependents_of(self, node_id: str) -> List[CompetencyEdge]:
        return list(self._edges.get(node_id, []))

    # ------------------------------------------------------------------
    def ready_nodes(self, mastered: Iterable[str]) -> List[CompetencyNode]:
        """Return nodes whose prerequisites are all mastered."""

        mastered_set: Set[str] = set(mastered)
        available: List[CompetencyNode] = []
        for node_id, node in self._nodes.items():
            prereqs = {edge.source for edge in self.dependencies_of(node_id) if edge.relation == "prerequisite"}
            if prereqs.issubset(mastered_set) and node_id not in mastered_set:
                available.append(node)
        return available

    # ------------------------------------------------------------------
    def pathway(self, start_nodes: Iterable[str]) -> List[CompetencyNode]:
        """Compute an ordered pathway using weighted breadth-first traversal."""

        visited: Set[str] = set()
        queue: List[Tuple[float, str]] = [(0.0, node_id) for node_id in start_nodes]
        ordered: List[CompetencyNode] = []
        while queue:
            queue.sort(key=lambda item: item[0])
            weight, node_id = queue.pop(0)
            if node_id in visited or node_id not in self._nodes:
                continue
            visited.add(node_id)
            ordered.append(self._nodes[node_id])
            for edge in self.dependents_of(node_id):
                queue.append((weight + edge.weight, edge.target))
        return ordered

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {node_id: self._serialize_node(node) for node_id, node in self._nodes.items()},
            "edges": [edge.to_dict() for edges in self._edges.values() for edge in edges],
            "modules": {
                name: {
                    "nodes": sorted(list(nodes)),
                    "edges": [
                        {
                            "source": src,
                            "target": tgt,
                            "relation": rel,
                        }
                        for (src, tgt, rel) in sorted(edge_refs)
                    ],
                }
                for name, nodes in self._modules.items()
                for edge_refs in [self._module_edges.get(name, set())]
            },
            "resources": {
                node_id: [res.to_dict() for res in resources]
                for node_id, resources in self._resources.items()
                if resources
            },
        }

    # ------------------------------------------------------------------
    def save_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "KnowledgeGraph":
        graph = cls()
        for node_id, data in payload.get("nodes", {}).items():
            node = CompetencyNode(**data)
            graph.add_node(node)
        for edge_data in payload.get("edges", []):
            graph.add_edge(CompetencyEdge(**edge_data))
        for name, module_payload in payload.get("modules", {}).items():
            nodes = set(module_payload.get("nodes", []))
            graph._modules[name] = {node for node in nodes if node in graph._nodes}
            edge_refs: Set[Tuple[str, str, str]] = set()
            for edge_info in module_payload.get("edges", []):
                src = edge_info.get("source")
                tgt = edge_info.get("target")
                relation = edge_info.get("relation", "prerequisite")
                if src in graph._nodes and tgt in graph._nodes:
                    edge_refs.add((src, tgt, relation))
            if edge_refs:
                graph._module_edges[name] = edge_refs
        for node_id, resources in payload.get("resources", {}).items():
            if node_id not in graph._nodes:
                continue
            for resource_payload in resources:
                try:
                    resource = ContentResource(**resource_payload)
                    graph.link_resource(node_id, resource)
                except Exception:  # pragma: no cover - corrupted payloads ignored
                    continue
        return graph

    # ------------------------------------------------------------------
    @classmethod
    def load_json(cls, path: Path) -> "KnowledgeGraph":
        payload = json.loads(path.read_text())
        return cls.from_dict(payload)

    # ------------------------------------------------------------------
    @staticmethod
    def _serialize_node(node: CompetencyNode) -> Dict[str, Any]:
        return {
            "domain": node.domain,
            "skill_id": node.skill_id,
            "label": node.label,
            "bloom_level": node.bloom_level,
            "proficiency_level": node.proficiency_level,
            "hsk_level": node.hsk_level,
            "metadata": node.metadata,
        }


def register_bpmn_modules(
    graph: KnowledgeGraph,
    *,
    matrix_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the BPMN skill matrix and register modules and resources.

    The helper keeps the knowledge graph aligned with the dedicated BPMN
    competency matrix by adding nodes for Symbolkenntnis,
    Modellierungsregeln und Fehleranalyse, wiring their sequencing and
    attaching curated learning resources (Diagrammeditor, animierte
    Abläufe, Process-Mining-Notebook).
    """

    resolved_path = (
        Path(matrix_path)
        if matrix_path is not None
        else Path(__file__).resolve().parent / "competencies" / "bpmn.skillmatrix.json"
    )
    try:
        payload = _load_matrix_payload(Path(resolved_path))
    except FileNotFoundError as exc:  # pragma: no cover - configuration guard
        raise FileNotFoundError(
            f"BPMN skill matrix not found at {resolved_path}"
        ) from exc
    except ValueError as exc:  # pragma: no cover - configuration guard
        raise ValueError(f"Invalid BPMN skill matrix: {resolved_path}") from exc

    domain = payload.get("domain", "bpmn")
    nodes_by_skill: Dict[str, CompetencyNode] = {}
    modules: Dict[str, List[CompetencyNode]] = {}
    for skill in payload.get("skills", []):
        skill_id = skill.get("id")
        if not skill_id:
            continue
        module_name = skill.get("module") or f"{domain}_module"
        metadata = {
            "focus": skill.get("focus", []),
            "attributes": skill.get("attributes", {}),
            "example_tasks": skill.get("example_tasks", []),
            "module": module_name,
            "version": payload.get("version"),
            "updated_at": payload.get("updated_at"),
        }
        metadata = {key: value for key, value in metadata.items() if value}
        node = CompetencyNode(
            domain=domain,
            skill_id=skill_id,
            label=skill.get("name", skill_id),
            bloom_level=skill.get("bloom_level", "K1"),
            metadata=metadata,
        )
        nodes_by_skill[skill_id] = node
        modules.setdefault(module_name, []).append(node)

    module_lookup: Dict[str, str] = {
        node.identifier: module for module, members in modules.items() for node in members
    }

    module_edges: Dict[str, List[CompetencyEdge]] = {name: [] for name in modules}
    cross_edges: List[CompetencyEdge] = []
    for relation in payload.get("sequencing", []):
        source_skill = relation.get("from")
        target_skill = relation.get("to")
        if not source_skill or not target_skill:
            continue
        source_node = nodes_by_skill.get(source_skill)
        target_node = nodes_by_skill.get(target_skill)
        if not source_node or not target_node:
            continue
        edge = CompetencyEdge(
            source=source_node.identifier,
            target=target_node.identifier,
            relation=relation.get("relation", "prerequisite"),
            weight=float(relation.get("weight", 1.0)),
        )
        source_module = module_lookup.get(edge.source)
        target_module = module_lookup.get(edge.target)
        if source_module and source_module == target_module:
            module_edges.setdefault(source_module, []).append(edge)
        else:
            cross_edges.append(edge)

    for module_name, module_nodes in modules.items():
        graph.add_module(module_name, module_nodes, module_edges.get(module_name, []))

    for edge in cross_edges:
        try:
            graph.add_edge(edge)
        except KeyError:  # pragma: no cover - nodes should exist already
            continue

    for resource_block in payload.get("resources", []):
        skill_id = resource_block.get("skill_id")
        if not skill_id or skill_id not in nodes_by_skill:
            continue
        node_identifier = nodes_by_skill[skill_id].identifier
        for item in resource_block.get("items", []):
            resource_id = item.get("resource_id")
            title = item.get("title")
            uri = item.get("uri")
            if not resource_id or not title or not uri:
                continue
            modality = item.get("modality", "other")
            metadata = {
                key: value
                for key, value in item.items()
                if key not in {"resource_id", "title", "uri", "modality"} and value
            }
            metadata.setdefault("media_cluster", "diagram_tools")
            resource = ContentResource(
                resource_id=resource_id,
                title=title,
                uri=uri,
                modality=modality,
                media_cluster="diagram_tools",
                metadata=metadata,
            )
            try:
                graph.link_resource(node_identifier, resource)
            except KeyError:  # pragma: no cover - safety guard
                continue

    return {
        "domain": domain,
        "modules": {
            module: sorted(node.identifier for node in members)
            for module, members in modules.items()
        },
        "edges_registered": len(payload.get("sequencing", [])),
    }


def register_math_modules(
    graph: KnowledgeGraph,
    *,
    matrix_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the math skill matrix and wire misconceptions plus applications."""

    resolved_path = (
        Path(matrix_path)
        if matrix_path is not None
        else Path(__file__).resolve().parent / "competencies" / "math.skillmatrix.json"
    )
    try:
        payload = _load_matrix_payload(Path(resolved_path))
    except FileNotFoundError as exc:  # pragma: no cover - configuration guard
        raise FileNotFoundError(f"Math skill matrix not found at {resolved_path}") from exc
    except ValueError as exc:  # pragma: no cover - configuration guard
        raise ValueError(f"Invalid math skill matrix: {resolved_path}") from exc

    domain = payload.get("domain", "math")
    module_nodes: Dict[str, Dict[str, CompetencyNode]] = {}
    module_edges: Dict[str, List[CompetencyEdge]] = {}
    skill_lookup: Dict[str, CompetencyNode] = {}
    pending_resources: List[Tuple[str, ContentResource]] = []

    summary = {
        "modules_registered": 0,
        "nodes_added": 0,
        "resources_linked": 0,
        "misconceptions_added": 0,
        "applications_added": 0,
    }

    for skill in payload.get("skills", []):
        skill_id = skill.get("id")
        if not skill_id:
            continue
        module_name = skill.get("module") or f"{domain}_module"
        metadata = {
            "difficulty": skill.get("difficulty"),
            "recommended_methods": skill.get("recommended_methods", []),
            "module": module_name,
        }
        metadata = {key: value for key, value in metadata.items() if value}
        node = CompetencyNode(
            domain=domain,
            skill_id=skill_id,
            label=skill.get("name", skill_id),
            bloom_level=skill.get("bloom_level", "K1"),
            metadata=metadata,
        )
        module_nodes.setdefault(module_name, {})[node.identifier] = node
        skill_lookup[skill_id] = node

        for subskill in skill.get("subskills", []):
            sub_id = subskill.get("id")
            if not sub_id:
                continue
            sub_node = CompetencyNode(
                domain=domain,
                skill_id=sub_id,
                label=subskill.get("name", sub_id),
                bloom_level=subskill.get("bloom_level", node.bloom_level),
                metadata={
                    "type": "subskill",
                    "parent_skill": skill_id,
                    "recommended_methods": skill.get("recommended_methods", []),
                },
            )
            module_nodes[module_name][sub_node.identifier] = sub_node
            module_edges.setdefault(module_name, []).append(
                CompetencyEdge(
                    source=node.identifier,
                    target=sub_node.identifier,
                    relation="decomposes_into",
                    weight=1.0,
                )
            )

        for idx, misconception in enumerate(skill.get("misconceptions", []), start=1):
            mis_id = misconception.get("id") or f"{skill_id}.misconception.{idx}"
            mis_node = CompetencyNode(
                domain=domain,
                skill_id=mis_id,
                label=misconception.get("description", mis_id),
                bloom_level=skill.get("bloom_level", "K2"),
                metadata={
                    "type": "misconception",
                    "parent_skill": skill_id,
                    "tags": misconception.get("tags", []),
                },
            )
            module_nodes[module_name][mis_node.identifier] = mis_node
            module_edges.setdefault(module_name, []).append(
                CompetencyEdge(
                    source=node.identifier,
                    target=mis_node.identifier,
                    relation="addresses_misconception",
                    weight=1.0,
                )
            )
            summary["misconceptions_added"] += 1

        for idx, application in enumerate(skill.get("practice_applications", []), start=1):
            app_id = application.get("id") or f"{skill_id}.application.{idx}"
            app_node = CompetencyNode(
                domain=domain,
                skill_id=app_id,
                label=application.get("description", app_id),
                bloom_level=skill.get("bloom_level", "K2"),
                metadata={
                    "type": "application",
                    "parent_skill": skill_id,
                },
            )
            module_nodes[module_name][app_node.identifier] = app_node
            module_edges.setdefault(module_name, []).append(
                CompetencyEdge(
                    source=node.identifier,
                    target=app_node.identifier,
                    relation="applies_in",
                    weight=1.0,
                )
            )
            summary["applications_added"] += 1

        for resource in skill.get("resources", []):
            resource_id = resource.get("resource_id")
            if not resource_id:
                continue
            metadata = {
                "description": resource.get("description"),
                "skill": skill_id,
            }
            metadata = {k: v for k, v in metadata.items() if v is not None}
            metadata.setdefault("media_cluster", "plotter")
            content = ContentResource(
                resource_id=resource_id,
                title=resource.get("title", resource_id),
                uri=resource.get("uri", ""),
                modality=resource.get("modality", "resource"),
                media_cluster="plotter",
                metadata=metadata,
            )
            pending_resources.append((node.identifier, content))
            summary["resources_linked"] += 1

    sequencing_edges: List[CompetencyEdge] = []
    for relation in payload.get("sequencing", []):
        source_skill = relation.get("from")
        target_skill = relation.get("to")
        if not source_skill or not target_skill:
            continue
        source_node = skill_lookup.get(source_skill)
        target_node = skill_lookup.get(target_skill)
        if not source_node or not target_node:
            continue
        sequencing_edges.append(
            CompetencyEdge(
                source=source_node.identifier,
                target=target_node.identifier,
                relation=relation.get("relation", "prerequisite"),
                weight=float(relation.get("weight", 1.0)),
            )
        )

    for module_name, nodes in module_nodes.items():
        edges = module_edges.get(module_name, [])
        graph.add_module(module_name, list(nodes.values()), edges)
        summary["modules_registered"] += 1
        summary["nodes_added"] += len(nodes)

    for edge in sequencing_edges:
        try:
            graph.add_edge(edge)
        except KeyError:  # pragma: no cover - nodes should exist already
            continue

    for node_identifier, resource in pending_resources:
        try:
            graph.link_resource(node_identifier, resource)
        except KeyError:  # pragma: no cover - safety guard
            continue

    return summary


def register_language_modules(
    graph: KnowledgeGraph,
    *,
    matrix_paths: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    """Load language skill matrices (JSON or YAML) and register their modules."""

    default_root = Path(__file__).resolve().parent / "competencies"
    if matrix_paths is None:
        candidates = list(default_root.glob("language.*.json"))
        candidates.extend(default_root.glob("language.*.yaml"))
        candidates.extend(default_root.glob("language.*.yml"))
        matrix_paths = sorted({Path(path).resolve() for path in candidates})

    summary: Dict[str, Any] = {
        "domains": {},
        "matrices_loaded": 0,
        "modules_registered": 0,
        "resources_linked": 0,
    }

    for matrix_path in matrix_paths or []:
        try:
            payload = _load_matrix_payload(Path(matrix_path))
        except FileNotFoundError as exc:  # pragma: no cover - configuration guard
            raise FileNotFoundError(f"Language skill matrix not found: {matrix_path}") from exc
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"Invalid language skill matrix: {matrix_path}") from exc

        domain = payload.get("domain", "language")
        nodes_by_skill: Dict[str, CompetencyNode] = {}
        modules: Dict[str, List[CompetencyNode]] = defaultdict(list)
        module_edges: Dict[str, List[CompetencyEdge]] = defaultdict(list)

        pending_resources: List[Tuple[str, ContentResource]] = []

        for skill in payload.get("skills", []):
            skill_id = skill.get("id")
            if not skill_id:
                continue
            module_name = skill.get("module") or f"{domain}_module"
            metadata = {
                "focus": skill.get("focus", []),
                "attributes": skill.get("attributes", {}),
                "example_tasks": skill.get("example_tasks", []),
                "module": module_name,
                "version": payload.get("version"),
                "updated_at": payload.get("updated_at"),
            }
            metadata = {key: value for key, value in metadata.items() if value}
            node = CompetencyNode(
                domain=domain,
                skill_id=skill_id,
                label=skill.get("name", skill_id),
                bloom_level=skill.get("bloom_level", "K1"),
                metadata=metadata,
            )
            nodes_by_skill[skill_id] = node
            modules[module_name].append(node)

            for subskill in skill.get("subskills", []):
                sub_id = subskill.get("id")
                if not sub_id:
                    continue
                sub_node = CompetencyNode(
                    domain=domain,
                    skill_id=sub_id,
                    label=subskill.get("name", sub_id),
                    bloom_level=subskill.get("bloom_level", node.bloom_level),
                    metadata={
                        "type": "subskill",
                        "parent_skill": skill_id,
                        "module": module_name,
                    },
                )
                modules[module_name].append(sub_node)
                module_edges[module_name].append(
                    CompetencyEdge(
                        source=node.identifier,
                        target=sub_node.identifier,
                        relation="decomposes_into",
                        weight=1.0,
                    )
                )

            for resource in skill.get("resources", []):
                resource_id = resource.get("resource_id")
                if not resource_id:
                    continue
                metadata = {
                    key: value
                    for key, value in resource.items()
                    if key not in {"resource_id", "title", "uri", "modality"} and value is not None
                }
                metadata.setdefault("media_cluster", "audio")
                content = ContentResource(
                    resource_id=resource_id,
                    title=resource.get("title", resource_id),
                    uri=resource.get("uri", ""),
                    modality=resource.get("modality", "audio"),
                    media_cluster="audio",
                    metadata=metadata,
                )
                pending_resources.append((node.identifier, content))

        module_lookup = {
            node.identifier: module
            for module, members in modules.items()
            for node in members
            if isinstance(node, CompetencyNode)
        }

        cross_edges: List[CompetencyEdge] = []
        for relation in payload.get("sequencing", []):
            source_skill = relation.get("from")
            target_skill = relation.get("to")
            if not source_skill or not target_skill:
                continue
            source_node = nodes_by_skill.get(source_skill)
            target_node = nodes_by_skill.get(target_skill)
            if not source_node or not target_node:
                continue
            edge = CompetencyEdge(
                source=source_node.identifier,
                target=target_node.identifier,
                relation=relation.get("relation", "prerequisite"),
                weight=float(relation.get("weight", 1.0)),
            )
            source_module = module_lookup.get(edge.source)
            target_module = module_lookup.get(edge.target)
            if source_module and source_module == target_module:
                module_edges[source_module].append(edge)
            else:
                cross_edges.append(edge)

        for module_name, members in modules.items():
            graph.add_module(module_name, members, module_edges.get(module_name, []))
            summary["modules_registered"] += 1

        for edge in cross_edges:
            try:
                graph.add_edge(edge)
            except KeyError:  # pragma: no cover
                continue

        for resource_block in payload.get("resources", []):
            skill_id = resource_block.get("skill_id")
            node = nodes_by_skill.get(skill_id)
            if not node:
                continue
            for item in resource_block.get("items", []):
                resource_id = item.get("resource_id")
                title = item.get("title")
                uri = item.get("uri")
                if not resource_id or not title or not uri:
                    continue
                metadata = {
                    key: value
                    for key, value in item.items()
                    if key not in {"resource_id", "title", "uri", "modality"} and value is not None
                }
                metadata.setdefault("media_cluster", "audio")
                resource = ContentResource(
                    resource_id=resource_id,
                    title=title,
                    uri=uri,
                    modality=item.get("modality", "audio"),
                    media_cluster="audio",
                    metadata=metadata,
                )
                pending_resources.append((node.identifier, resource))

        for node_identifier, resource in pending_resources:
            try:
                graph.link_resource(node_identifier, resource)
                summary["resources_linked"] += 1
            except KeyError:  # pragma: no cover
                continue

        summary["matrices_loaded"] += 1
        summary["domains"][domain] = {
            "skills": len(nodes_by_skill),
            "modules": len(modules),
        }

    return summary


__all__ = [
    "CompetencyNode",
    "CompetencyEdge",
    "ContentResource",
    "KnowledgeGraph",
    "register_bpmn_modules",
    "register_math_modules",
    "register_language_modules",
]

