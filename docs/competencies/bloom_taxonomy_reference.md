# Bloom's Taxonomy Reference

This reference articulates the six cognitive levels of Bloom's revised taxonomy used across the adaptive learning pathways. It
complements the machine-readable definitions in [`bloom_levels.json`](../../bloom_levels.json) and should be used when designing
learning objectives, prompts, rubrics, and analytics aligned with Bloom levels.

| Level ID | Label | Cognitive Focus | Representative Action Verbs | Example Evidence Artifacts |
|----------|-------|-----------------|-----------------------------|-----------------------------|
| K1 | Remembering | Recall of facts, terms, and basic concepts. | define, list, recall, identify | Flashcard recall checks, vocabulary matching, fact quizzes. |
| K2 | Understanding | Demonstrate comprehension and explain ideas. | summarize, classify, describe, infer | Concept mapping exercises, short explanations, paraphrasing tasks. |
| K3 | Applying | Use knowledge or procedures in new contexts. | implement, calculate, solve, execute | Worked problem solutions, interactive simulations, role-play dialogues applying grammar. |
| K4 | Analyzing | Break down information, detect patterns, diagnose issues. | compare, differentiate, decompose, attribute | Error analysis worksheets, process mining diagnostics, text structure annotations. |
| K5 | Evaluating | Make judgements using criteria and evidence. | critique, justify, prioritize, recommend | Peer review rubrics, argumentative essays, optimisation rationales. |
| K6 | Creating | Assemble elements into a coherent or original whole. | design, compose, generate, formulate | BPMN redesign proposals, project-based assessments, original narratives or presentations. |

## Usage Guidelines

1. **Instructional Design**: map each learning objective to a Bloom level before generating activities or AI prompts. Ensure that
   recommended modalities match the depth of processing expected at that level.
2. **Assessment Calibration**: align scoring rubrics and analytics thresholds (see `bloom_levels.json:min_score`) with the level's
   mastery expectations. For example, mastery for K5 items should require substantiated justification, not just correct answers.
3. **Adaptive Logic**: when orchestrating paths, progress learners only after consistent evidence at the target level. Use the Bloom
   identifier (`K1`–`K6`) as part of the key in mastery stores and knowledge graph nodes.
4. **LLM Prompting**: incorporate the level label or action verbs into prompts to control syntactic complexity and reasoning depth.
5. **Reporting & Research**: cite this reference when documenting experiments or communicating the progression model to stakeholders.

Maintaining both the structured JSON and this human-readable reference keeps the taxonomy consistent across data pipelines,
instructional design tools, and qualitative reviews.
