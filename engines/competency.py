"""Competency mapping and domain-aware orchestration utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from knowledge_graph import (
    CompetencyNode,
    KnowledgeGraph,
    CompetencyEdge,
    register_bpmn_modules,
    register_language_modules,
    register_math_modules,
)


@dataclass
class CompetencyTemplate:
    """Describes expectations for a domain level."""

    domain: str
    level_id: str
    bloom_range: Tuple[str, str]
    description: str
    modalities: List[str]
    prompts: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillDefinition:
    """Atomic skill aligned to competencies and knowledge graph node."""

    skill_id: str
    label: str
    bloom_level: str
    proficiency_level: Optional[str] = None
    hsk_level: Optional[int] = None
    domain: str = "generic"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_node(self) -> CompetencyNode:
        return CompetencyNode(
            domain=self.domain,
            skill_id=self.skill_id,
            label=self.label,
            bloom_level=self.bloom_level,
            proficiency_level=self.proficiency_level,
            hsk_level=self.hsk_level,
            metadata=self.metadata,
        )


# ---------------------------------------------------------------------------
# Domain competency templates
# ---------------------------------------------------------------------------

BUSINESS_TEMPLATES: List[CompetencyTemplate] = [
    CompetencyTemplate(
        domain="business_process",
        level_id="awareness",
        bloom_range=("K1", "K3"),
        description="Identify stakeholders and high-level process steps.",
        modalities=["video", "infographic", "quiz"],
        prompts={
            "activity": (
                "You are a BPM coach. Create a scenario-based reflection question"
                " focusing on stakeholder identification at Bloom {bloom_level}."
            ),
            "feedback": "Provide actionable suggestions referencing BPMN best practices.",
        },
    ),
    CompetencyTemplate(
        domain="business_process",
        level_id="modeling",
        bloom_range=("K3", "K5"),
        description="Model processes with BPMN constructs and analyse flow.",
        modalities=["interactive", "simulation", "case_study"],
        prompts={
            "activity": (
                "Design an interactive BPMN modelling task that emphasises {lexical_focus}"
                " terminology and requires gateway reasoning."
            ),
            "feedback": "Return annotated BPMN diagram critique.",
        },
    ),
    CompetencyTemplate(
        domain="business_process",
        level_id="optimization",
        bloom_range=("K4", "K6"),
        description="Optimise processes using event logs and KPIs.",
        modalities=["process_mining", "workshop", "simulation"],
        prompts={
            "activity": "Generate an optimisation challenge using process mining metrics.",
            "feedback": "Summarise recommended improvements referencing throughput and SLA.",
        },
    ),
]

MATHEMATICS_TEMPLATES: List[CompetencyTemplate] = [
    CompetencyTemplate(
        domain="mathematics",
        level_id="foundational",
        bloom_range=("K1", "K2"),
        description="Build fluency with arithmetic facts and representations.",
        modalities=["interactive", "manipulative", "video"],
        prompts={
            "activity": (
                "Pose a concrete arithmetic fluency drill with {step_count} scaffolded hints"
                " and numbers within {numerical_range}."
            ),
            "feedback": "Offer growth-mindset encouragement and concrete error fixes.",
        },
    ),
    CompetencyTemplate(
        domain="mathematics",
        level_id="conceptual",
        bloom_range=("K2", "K4"),
        description="Develop conceptual understanding and reasoning.",
        modalities=["socratic_dialogue", "proof_sketch", "visual"],
        prompts={
            "activity": "Craft a Socratic dialogue that elicits conceptual reasoning steps.",
            "feedback": "Summarise misconceptions and next-step prompts.",
        },
    ),
    CompetencyTemplate(
        domain="mathematics",
        level_id="applied",
        bloom_range=("K4", "K6"),
        description="Apply mathematics to modelling and problem solving contexts.",
        modalities=["project", "simulation", "data_story"],
        prompts={
            "activity": "Generate a modelling challenge with authentic data context.",
            "feedback": "Highlight modelling assumptions and reflect on solution quality.",
        },
    ),
]

LANGUAGE_TEMPLATES: List[CompetencyTemplate] = [
    CompetencyTemplate(
        domain="language_de_en",
        level_id="beginner",
        bloom_range=("K1", "K2"),
        description="Survival phrases, introductions, and everyday vocabulary.",
        modalities=["dialogue", "flashcards", "listening"],
        prompts={
            "activity": "Compose a bilingual role-play focusing on greetings and family.",
            "feedback": "Provide pronunciation tips and literal translations.",
        },
        metadata={"hsk_level": 1},
    ),
    CompetencyTemplate(
        domain="language_de_en",
        level_id="intermediate",
        bloom_range=("K2", "K4"),
        description="Narrate past events and discuss preferences.",
        modalities=["story", "audio", "debate"],
        prompts={
            "activity": "Create a short dialogue about hobbies using separable verbs.",
            "feedback": "Highlight word order and case usage.",
        },
        metadata={"hsk_level": 3},
    ),
    CompetencyTemplate(
        domain="language_zh_en",
        level_id="hsk1",
        bloom_range=("K1", "K2"),
        description="Introduce self, numbers, simple questions.",
        modalities=["dialogue", "flashcards", "listening"],
        prompts={
            "activity": "Generate dialogue using only HSK1 vocabulary: {word_bank}.",
            "feedback": "Correct tone usage and provide pinyin.",
        },
        metadata={"hsk_level": 1},
    ),
    CompetencyTemplate(
        domain="language_zh_en",
        level_id="hsk3",
        bloom_range=("K2", "K4"),
        description="Describe daily routines, express opinions with simple clauses.",
        modalities=["dialogue", "video_script", "interactive"],
        prompts={
            "activity": "Write a conversation requiring sequential markers and measure words.",
            "feedback": "Explain grammar patterns and provide spaced repetition schedule.",
        },
        metadata={"hsk_level": 3},
    ),
    CompetencyTemplate(
        domain="language_zh_en",
        level_id="hsk5",
        bloom_range=("K3", "K5"),
        description="Discuss abstract topics and infer intent in authentic materials.",
        modalities=["debate", "article", "presentation"],
        prompts={
            "activity": "Produce an argumentative text including idiomatic expressions.",
            "feedback": "Analyse rhetorical devices and suggest advanced synonyms.",
        },
        metadata={"hsk_level": 5},
    ),
]

ALL_TEMPLATES: Dict[str, List[CompetencyTemplate]] = {
    "business_process": BUSINESS_TEMPLATES,
    "mathematics": MATHEMATICS_TEMPLATES,
    "language_de_en": [tpl for tpl in LANGUAGE_TEMPLATES if tpl.domain == "language_de_en"],
    "language_zh_en": [tpl for tpl in LANGUAGE_TEMPLATES if tpl.domain == "language_zh_en"],
}


# ---------------------------------------------------------------------------
# Skill registry
# ---------------------------------------------------------------------------

LANGUAGE_SKILLS: List[SkillDefinition] = [
    SkillDefinition(
        domain="language_zh_en",
        skill_id="vocab_hsk1_family",
        label="Introduce family members",
        bloom_level="K1",
        proficiency_level="HSK1",
        hsk_level=1,
        metadata={"word_bank": ["妈妈", "爸爸", "哥哥", "姐姐", "家"]},
    ),
    SkillDefinition(
        domain="language_zh_en",
        skill_id="grammar_hsk3_aspect",
        label="Use aspect particles 了 and 过",
        bloom_level="K3",
        proficiency_level="HSK3",
        hsk_level=3,
        metadata={"patterns": ["吃了", "去过", "没……过"]},
    ),
    SkillDefinition(
        domain="language_de_en",
        skill_id="grammar_cases_basic",
        label="Apply nominative and accusative cases",
        bloom_level="K2",
        proficiency_level="A2",
        metadata={"structures": ["der", "den", "die", "ein"]},
    ),
]

MATHEMATICS_SKILLS: List[SkillDefinition] = [
    SkillDefinition(
        domain="mathematics",
        skill_id="arith_fluency_addition",
        label="Add within 100",
        bloom_level="K1",
        metadata={"numerical_range": [0, 100]},
    ),
    SkillDefinition(
        domain="mathematics",
        skill_id="algebra_quadratic_factor",
        label="Factor quadratic trinomials",
        bloom_level="K3",
        metadata={"form": "ax^2+bx+c"},
    ),
    SkillDefinition(
        domain="mathematics",
        skill_id="modelling_linear_programming",
        label="Model linear optimisation problems",
        bloom_level="K5",
        metadata={"context": "resource allocation"},
    ),
]

BUSINESS_SKILLS: List[SkillDefinition] = [
    SkillDefinition(
        domain="business_process",
        skill_id="bpmn_identify_events",
        label="Identify BPMN events",
        bloom_level="K2",
        metadata={"lexical_focus": ["start event", "end event"]},
    ),
    SkillDefinition(
        domain="business_process",
        skill_id="bpmn_model_gateways",
        label="Model decision gateways",
        bloom_level="K3",
        metadata={"lexical_focus": ["exclusive", "parallel", "inclusive"]},
    ),
    SkillDefinition(
        domain="business_process",
        skill_id="process_mining_variants",
        label="Analyse process variants",
        bloom_level="K5",
        metadata={"metrics": ["throughput", "conformance"]},
    ),
]

SKILL_REGISTRY: Dict[str, List[SkillDefinition]] = {
    "language_zh_en": [skill for skill in LANGUAGE_SKILLS if skill.domain == "language_zh_en"],
    "language_de_en": [skill for skill in LANGUAGE_SKILLS if skill.domain == "language_de_en"],
    "mathematics": MATHEMATICS_SKILLS,
    "business_process": BUSINESS_SKILLS,
}


def build_knowledge_graph() -> KnowledgeGraph:
    """Construct a starter knowledge graph across domains."""

    graph = KnowledgeGraph()
    register_bpmn_modules(graph)
    register_math_modules(graph)
    register_language_modules(graph)

    for skills in SKILL_REGISTRY.values():
        for skill in skills:
            node = skill.to_node()
            if graph.get_node(node.identifier):
                continue
            graph.add_node(node)

    dependencies = [
        ("business_process:bpmn_identify_events:K2:-:-", "business_process:bpmn_model_gateways:K3:-:-"),
        ("business_process:bpmn_model_gateways:K3:-:-", "business_process:process_mining_variants:K5:-:-"),
        ("mathematics:arith_fluency_addition:K1:-:-", "mathematics:algebra_quadratic_factor:K3:-:-"),
        ("mathematics:algebra_quadratic_factor:K3:-:-", "mathematics:modelling_linear_programming:K5:-:-"),
        ("language_zh_en:vocab_hsk1_family:K1:HSK1:HSK1", "language_zh_en:grammar_hsk3_aspect:K3:HSK3:HSK3"),
    ]
    for source, target in dependencies:
        try:
            graph.add_edge(
                CompetencyEdge(
                    source=source,
                    target=target,
                    relation="prerequisite",
                    weight=1.0,
                )
            )
        except KeyError:
            continue
    return graph


@dataclass
class Recommendation:
    """Represents a multimodal learning recommendation."""

    competency_id: str
    modality: str
    prompt: str
    feedback_prompt: str
    metadata: Dict[str, Any]


__all__ = [
    "CompetencyTemplate",
    "SkillDefinition",
    "Recommendation",
    "build_knowledge_graph",
    "SKILL_REGISTRY",
    "ALL_TEMPLATES",
]

