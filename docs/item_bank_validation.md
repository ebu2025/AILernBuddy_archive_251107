# Item Bank Validation Guarantees

This note documents the validation checks enforced by `item_bank.ItemBank._load`
and the regression coverage present in `tests/test_item_bank.py`. Stakeholders can
use it as the authoritative reference for what the loader currently guarantees.

## Loader Guarantees (`item_bank.ItemBank._load`)

The loader validates both the shape of the item bank file and every entry inside
it. Attempts to load an invalid bank raise `ItemValidationError` (or
`FileNotFoundError` when the file is missing) before any partial state is
committed. The checks are applied in the following order:

1. **File existence and top-level structure**
   - The JSON file must exist.
   - The root JSON value must be a list.
   - Each element in the list must be an object/dict.
2. **Required fields**
   - Every item must include non-empty values for: `id`, `domain`, `skill_id`,
     `bloom_level`, `stimulus`, and `elo_target`.
   - Item identifiers are coerced to strings and must be unique across the file.
3. **Optional metadata containers**
   - `metadata`, when provided, must be a JSON object (defaults to `{}`).
   - `references`, when provided, must be a list (defaults to `[]`).
4. **Exposure limits**
   - `exposure_limit`, when present, must parse to a non-negative integer.
5. **Difficulty & target scores**
   - `difficulty`, when present, must parse to a floating-point number.
   - `elo_target` must parse to a floating-point number.
6. **Answer key or rubric**
   - Each item must provide at least one of `answer_key` or `rubric_id`.
7. **Multiple-choice enrichments (within `metadata`)**
   - If `metadata["choices"]` exists, it must be a non-empty list.
   - `metadata["distractor_rationales"]` must be a non-empty dict covering every
     distractor option (all choices except the `answer_key`).
   - Each rationale must be a non-empty string.
   - When the choices block passes validation, the loader defaults
     `metadata["item_type"]` to `"mcq"`.
8. **Normalized storage**
   - On success each validated item is normalized (string conversions, default
     containers, parsed numeric values) before being stored in-memory and, when
     `auto_sync=True`, persisted via `db.upsert_item_bank_entries`.

## Current Test Coverage (`tests/test_item_bank.py`)

The regression tests cover both the happy path and key error scenarios:

- **`test_item_bank_loads_and_syncs`** – Verifies that a well-formed JSON file is
  loaded, exposes two items via `ItemBank.items`, and triggers the database sync
  path via `db.list_item_bank`.
- **`test_item_bank_exposure_limit`** – Exercises the selection logic with
  `exposure_limit` constraints to ensure loader-parsed limits integrate with the
  exposure tracking subsystem.
- **`test_item_bank_validation`** – Confirms that malformed items (missing
  required fields) raise `ItemValidationError`, preventing invalid data from
  entering the bank.

Future validation rules should extend these tests (or add new ones) so that the
coverage grows in lockstep with any additional loader safeguards.
