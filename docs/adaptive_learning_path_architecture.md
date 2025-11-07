# Adaptive Learning Path Architecture

## Vision

Create a dynamic, competency-aligned learning experience that continuously adapts to learner signals across business process management, mathematics, and bilingual language learning (German↔English, English↔Chinese). Static syllabi are replaced by responsive pathways governed by evidence, competency depth, and knowledge dependencies.

## Multi-dimensional Competency Model

| Dimension | Description | Data Sources |
|-----------|-------------|-------------|
| Skill Matrix | Domain-specific atomic skills (e.g., "Model BPMN Gateways", "Solve quadratic equations", "Describe family members in German"). | Item metadata, manual tagging, knowledge graph nodes. |
| Bloom Level | Cognitive demand K1–K6. Automatically inferred from activity metadata and assessment rubrics. | Bloom catalogue (`bloom_levels.json`) + [human-readable reference](competencies/bloom_taxonomy_reference.md). |
| Proficiency / HSK Level | Numeric indicator (HSK 1–6) for Chinese vocabulary and grammar, mapped to CEFR-like bands for other languages. | `competency.language.hsk.yaml` (new asset) + lexical frequency lists. |
| Process Competency | BPM maturity stage (Awareness → Optimization). | BPMN process mining logs, business simulation outcomes. |

Learner state stores mastery scores per skill and level, along with contextual metrics (latency, confidence, recency, modality preference).

## Domain-specific Level Templates

Each domain defines templates describing the competency expectations, vocabulary/notation limits, canonical activity types, and recommended multimodal assets.

- **Business Process Management**
  - *Level 1 (Awareness)*: Identify process stakeholders; consume infographic/video explainer.
  - *Level 2 (Modeling)*: Construct BPMN diagrams with tasks and gateways using interactive canvas.
  - *Level 3 (Optimization)*: Analyze event logs, suggest improvements; use process mining sandbox.

- **Mathematics**
  - *Foundational*: Arithmetic fluency, visual number talks, automatic scaffolding.
  - *Conceptual*: Derive formulas, manipulate algebraic expressions, Socratic LLM dialogues.
  - *Applied*: Multi-step modeling problems, spreadsheet simulations, proof sketches.

- **Language (German↔English, English↔Chinese)**
  - Align grammar/vocabulary to HSK or CEFR bands.
  - Provide explicit word lists per level with part-of-speech tags and difficulty metadata.
  - Offer prompt templates for LLM activity generation (see `docs/llm_prompt_library.md`).

## Adaptive Progression Engine

1. **Signals**: Assessment results, engagement analytics, process mining metrics, learner feedback.
2. **Competency Projection**: Map each signal to relevant nodes in the competency matrix.
3. **Decision Logic**:
   - Bloom-level controller (existing `AdaptiveLearningPathManager`).
   - Domain adapter (new `DomainAdaptiveOrchestrator`) modifies promotion thresholds and modality recommendations based on competency type.
4. **Recommendation Output**: Next learning experience, feedback tone, resources, and knowledge graph updates.

## Knowledge Graph & Dependencies

- Nodes: `CompetencyNode` representing skill + level combination.
- Edges: Prerequisites, co-requisites, or reinforcement loops (weighted).
- Stored in `knowledge_graph.py` with JSON serialization for auditability.
- Used to assemble personalized paths by topological sorting constrained by learner mastery.

## BPMN-Based Learning Processes

BPMN models capture sequential and parallel skill development. Process variants per domain are authored under `process_models/` using BPMN 2.0 XML (see `process_models/language_dual_track.bpmn`). Parallel gateways orchestrate vocabulary + grammar practice; event subprocesses capture remediation loops triggered by low confidence.

## Multimodal Content Selection

Content selection algorithm ranks modalities by:
- Competency type (e.g., BPM modeling favors interactive diagrams).
- Learner preference profile (video vs. text).
- Experiment outcomes (A/B test uplift stored in experimentation registry).

## Experimentation & Optimization Workflow

1. Simulate learner personas (novice, fast-advancing, struggling) using Monte Carlo sampling in `engines/simulation.py`.
2. Run A/B tests comparing progression policies; metrics include time-to-mastery, retention, satisfaction proxies.
3. Feed winning variants back into orchestration parameters.
4. Document iterations in `docs/experiments/` for transparency and research dissemination.

## Continuous Improvement Loop

- **Instrumentation**: xAPI statements + database logs.
- **Process Mining**: Derive high-performing traces, update knowledge graph edge weights.
- **LLM Prompt Refinement**: Evaluate lexical diversity and syntactic complexity per level, adjust prompt templates to achieve desired distributions.
- **Governance**: Quarterly review of competency mappings with SMEs; automated regression tests on simulation scenarios.

