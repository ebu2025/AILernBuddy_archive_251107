# Language Strategy Registry and Template Formats

## Current HSK/CEFR Implementation Overview

The text generation engine maintains a registry of available language strategy
packages inside `engines/text_generation.py`. The sections below catalogue how
HSK (Mandarin → English) and CEFR (German → English) templates flow through the
registry today:

- `_build_language_strategies` is the single loader for strategy metadata. It
  looks for JSON payloads under `data/language/` and registers them by
  `language_pair`. The current inventory maps `zh_en` to the HSK library at
  `data/language/hsk_levels.json` and `de_en` to the CEFR library at
  `data/language/cefr_de_en_templates.json`. Each registry entry captures the
  `strategy` name (`"hsk"` or `"cefr"`), the nested `levels` dictionary, and
  `metadata` with the source path so downstream consumers can trace the
  originating template.【F:engines/text_generation.py†L33-L64】
- `_select_language_level` enforces that a request references a registered
  `language_pair` and an available `skill_level`. It raises
  `UnknownLanguagePairError` or `UnknownSkillLevelError` when the registry lacks
  the requested combination, returning the strategy identifier, level payload,
  and metadata when the lookup succeeds.【F:engines/text_generation.py†L76-L89】
- `TextGenerationProgressionEngine.generate_activity` consumes the registry
  record to build deterministic payloads. For each `skill_level` it resolves the
  chosen lexicon mode (`"simple"` or `"broad"`), stitches together
  `text_tokens`, vocabulary lists, exercises, tasks, Bloom defaults, grammar
  focus, competencies, and the prompt suite, then logs a
  `text_generation` learning event tagged with the originating strategy and
  `language_pair`. The resulting activity includes the source metadata so the
  client can attribute content to the CEFR or HSK templates.【F:engines/text_generation.py†L91-L170】

These mechanics ensure that expanding the CEFR/HSK offering—or adding new
language pairs—requires only template files and optional helpers. The engine
code already supports any strategy following the expected schema.

## CEFR and HSK Template Inventory

Both template families live under `data/language/` and declare their
`language_pair` at the root of the JSON document (`"zh_en"` for HSK and
`"de_en"` for CEFR). Each file’s `levels` dictionary is keyed by the published
proficiency labels (e.g., `HSK1`, `HSK3`, `A2`, `B1`) and provides matched
`simple`/`broad` lexicon modes alongside a full prompt suite. Refer to
`data/language/hsk_levels.json` and `data/language/cefr_de_en_templates.json`
for concrete examples that follow the schema described below.

## Template JSON Format

Template files are JSON documents with the following structure (mirroring the
current HSK and CEFR payloads under `data/language/`):

```json
{
  "language_pair": "<source>_<target>",
  "levels": {
    "<LEVEL_ID>": {
      "default_bloom": "<Bloom code>",
      "grammar_focus": ["..."],
      "competencies": ["..."],
      "simple": {
        "text_tokens": ["..."],
        "vocab": [
          {
            "token": "...",
            "translation": "...",
            "bloom_level": "...",
            "hsk_level" | "cefr_level": "...",
            "notes": "..."
          }
        ],
        "exercises": [
          {
            "id": "...",
            "prompt": "...",
            "bloom_level": "...",
            "hsk_level" | "cefr_level": "...",
            "difficulty": "..."
          }
        ],
        "tasks": [
          {
            "id": "...",
            "description": "...",
            "target_bloom_level": "..."
          }
        ]
      },
      "broad": { "text_tokens": [...], "vocab": [...], "exercises": [...], "tasks": [...] },
      "prompts": {
        "listening_comprehension": {
          "instructions": "...",
          "audio": {
            "asset": "...",
            "duration_seconds": <number>,
            "transcript": "..."
          },
          "keywords": ["..."],
          "bloom_level": "..."
        },
        "translation": {
          "source_text": "...",
          "target_language": "...",
          "rubric": "...",
          "bloom_level": "...",
          "reference_translation": "..."
        },
        "dialogue_simulation": {
          "scenario": "...",
          "roles": ["..."],
          "bloom_level": "..."
        },
        "pronunciation": {
          "focus_syllables": ["..."],
          "target_tones" | "stress_pattern": [...],
          "model_audio": "..."
        }
      }
    }
  }
}
```

Key differences between HSK and CEFR templates lie in the level-specific
fields (`hsk_level` vs `cefr_level`) and pronunciation metadata (tones vs
stress pattern), but both share the same nesting so that the registry can treat
all templates uniformly.

## Extending the Registry

Work on WP4 should concentrate on augmenting this registry-driven model:

1. Contribute new JSON templates or helper utilities that plug into
   `_build_language_strategies` rather than reimplementing loader logic.
2. Provide schema validation or authoring aids that ensure new templates follow
   the structure above, including the prompt suite and lexicon modes.
3. Update the automated tests in `tests/test_text_generation_engine.py` whenever
   schema expectations change so contributors receive immediate feedback.

Following these principles keeps `TextGenerationProgressionEngine` stable while
allowing the library of language strategies to grow organically.
