# Adaptive Learning Path Experiments

This log tracks experiments, simulations, and A/B tests used to optimise the adaptive learning framework.

## Experiment Registry

| ID | Hypothesis | Variation(s) | Cohort | Metric(s) | Outcome | Next Step |
|----|------------|--------------|--------|-----------|---------|-----------|
| EXP-001 | Domain adapters outperform single-threshold Bloom controller. | A: Baseline (uniform thresholds) \| B: Competency-weighted thresholds. | Simulated novices/intermediates | Time-to-mastery, frustration proxy | Pending | Run Monte Carlo via `engines/simulation.py`. |
| EXP-002 | Vocabulary-constrained prompts improve lexical retention for HSK 2 learners. | A: Open vocabulary \| B: Word-bank constrained | 60 bilingual learners | Retention quiz, lexical diversity | Pending | Deploy content; monitor lexical metrics. |
| EXP-003 | Parallel grammar + speaking track (BPMN dual-track model) reduces plateau. | A: Sequential flow \| B: Parallel flow | Advanced Chinese learners | Level progression velocity | Pending | Instrument event subprocess completion time. |

## Simulation & Evaluation Setup

- **Personas**: `Novice`, `FastAdvancer`, `SlowConfidence`. Parameterised by initial mastery, confidence bias, response latency.
- **Policies**: Baseline vs. DomainAdaptiveOrchestrator.
- **Metrics**: Average sessions to level completion, dropout probability, knowledge graph coverage, modality utilisation.
- **Methods**:
  - *Monte Carlo sampling* via `LearningSimulation.run_monte_carlo` for stress-testing policy robustness across synthetic cohorts.
  - *Trace replay* via `LearningSimulation.run_trace_replay` using observed process mining logs to validate fidelity against real learner behaviour.
- **Guidance**: Monte Carlo alone may overfit to assumed priors; always triangulate results with replayed traces and sensitivity analyses on attempt models.

## Process Mining Workflow

1. Export event logs with columns: `user_id`, `competency_id`, `state`, `timestamp`.
2. Run discovery using PM4Py (offline). Import results as JSON edge weight updates via `knowledge_graph.update_edge_weight`.
3. Compare discovered models against canonical BPMN flows (`process_models/`).

## Iteration Notes

- Document prompt revisions and lexical analysis scripts in this file with dates.
- Use versioned sections (`### 2024-05-17`) for chronological tracking.

