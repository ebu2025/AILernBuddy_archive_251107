# LLM Prompt Library for Adaptive Content Generation

This library standardises prompts used to generate domain-specific activities with controllable difficulty. Prompts are parameterised to tune vocabulary breadth, syntactic complexity, and interaction style.

## Shared Prompt Controls

| Parameter | Description |
|-----------|-------------|
| `target_competency` | Reference to `CompetencyNode` identifier. |
| `bloom_level` | Bloom K-level guiding cognitive demand. |
| `proficiency_level` | HSK or CEFR level to constrain vocabulary/grammar. |
| `lexical_focus` | Word families or morphemes to emphasise. |
| `syntax_complexity` | Values: `simple`, `compound`, `complex`. |
| `modality` | `text`, `video_script`, `interactive`. |

Prompts are stored as JSON templates in `engines/competency.py` for reuse in orchestration.

## Business Process Management

```
You are a BPM coach helping learners at the {target_competency.label} stage.
Design a {modality} activity that requires learners to operate at Bloom level {bloom_level}.
Keep instructions concise (<150 words). Include realistic process artifacts and emphasise {lexical_focus} terminology.
```

- Difficulty modulation achieved via scenario complexity and number of process elements.
- Feedback channel instructs LLM to deliver model answers referencing BPMN best practices.

## Mathematics

```
Role: Socratic math tutor guiding {target_competency.label}.
Produce a {modality} learning interaction that demands Bloom level {bloom_level} thinking.
Constraints:
- Use numbers within {numerical_range}.
- Ensure exactly {step_count} reasoning steps.
- Provide adaptive hints triggered by incorrect attempts.
```

## Language: German↔English

### HSK-aligned Template

```
Generate a bilingual dialogue for HSK {proficiency_level} learners.
Vocabulary must be selected from the provided word list: {word_bank}.
Limit sentences to {syntax_complexity} structures.
Include comprehension questions that target Bloom level {bloom_level}.
```

### Vocabulary Expansion Prompt

```
Create 10 thematic vocabulary flashcards for topic {topic} at HSK {proficiency_level}.
Each entry: word, pronunciation, part of speech, mnemonic, example sentence (<=12 words).
```

## Language: English↔Chinese

- Provide translation direction parameter to enforce source/target language usage.
- Encourage cultural context embedding for advanced HSK levels.

## Prompt Evaluation Checklist

1. Validate lexical coverage using word-level classifiers.
2. Score syntactic complexity via dependency parse depth.
3. Human-in-the-loop sampling for tone appropriateness.

