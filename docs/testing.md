# Testing Conventions

This document inventories the fixtures and deterministic patterns currently relied on by the automated test suite. New tests should build on these shared helpers instead of introducing parallel infrastructure.

## Shared Fixtures

### `tests/conftest.py::temp_db`
- Creates a temporary SQLite database under the pytest-provided `tmp_path` and patches `db.DB_PATH` to point to it.
- Patches `xapi.DB_PATH` when the optional module is available so telemetry helpers write into the same temporary store.
- Calls `db.init()` to apply schema migrations before returning the temporary path as a string.
- Keeps database writes isolated per-test while preserving the production APIs exposed by `db` and `xapi`.

## Deterministic Test Patterns

### `tests/test_progression_engine.py`
- Uses `tempfile.TemporaryDirectory` inside the unittest-style `ProgressionEngineTests` to isolate `DB_PATH` via environment variables for engines that call straight into `db`.
- Relies on the canonical Bloom level ordering from `bloom_levels.BLOOM_LEVELS`. The test precomputes fallbacks (`_K_LEVEL_SEQUENCE`, `LOWEST_LEVEL`, `SECOND_LEVEL`) so assertions remain stable even if the configuration omits `k_level_sequence`.
- Seeds subject progress with `db.upsert_subject`, `ensure_progress_record`, and explicit `upsert_learning_path_state` payloads that reflect the production schema (`levels`, `current_level`, optional `history`).
- Exercises deterministic score windows (e.g., `[0.9, 0.85, 0.88]` for advancement or `[0.3, 0.28, 0.25]` for regression) so the resulting averages and reasons can be asserted exactly.
- Guards progression updates using explicit confidence thresholds, cooldown behaviour, and environment feature flags (e.g., setting `ENABLE_TD_BKT` to force TD-BKT code paths) while always restoring prior environment values.
- Validates persistence APIs (`log_learning_event`, `list_bloom_progress_history`, `save_assessment_result`) by inserting known payloads and verifying stored records, including hint plans and explanation strings.

### `tests/test_domain_adapter.py`
- Wraps the `DomainAdaptiveOrchestrator` in the shared `orchestrator` fixture, ensuring every test runs against the `temp_db` database path initialised by `db.init()`.
- Supplies deterministic payloads for BPMN, language, and mathematics assessments so downstream adapters create predictable step results and error patterns that can be asserted without mocking internals.
- Confirms domain-specific engine selection (e.g., `window_size == 4` for `business_process`) and fallback logic by comparing to the base `ProgressionEngine` type.
- Verifies modality prioritisation and feedback tone heuristics using fixed skill IDs and confidence/outcome tuples.

### `tests/test_chat_json_schema.py`
- Centralises monkeypatching in `_prepare_chat_environment`, which captures database interactions in an in-memory `recorded` dictionary while stubbing out `app.db` methods, learning path updates, and progression evaluation.
- Defines `_valid_response_payload()` as the canonical JSON structure emitted by the chat completion pipeline, mirroring `docs/json/README.md`. Tests reuse this payload to assert schema stability.
- Stubs `app.generate_with_continue` to yield deterministic newline-delimited responses, enabling explicit assertions on retry logic (`schema_retry` markers in `path_decisions`) and stored raw responses.
- Ensures progression evaluation gets invoked with `(user_id, topic, last_attempt_id)` tuples and that assessment results, follow-up state transitions, and learning path history updates are recorded in predictable order.
- When simulating failures, forces invalid JSON or schema shape so the code under test emits consistent HTTP errors and telemetry paths without persisting partial records.

## Extending the Suite

- Document new deterministic patterns (scoring windows, payload layouts, environment flags) in this file so contributors understand the shared expectations before modifying fixtures.
- Prefer extending `temp_db`, `_prepare_chat_environment`, or the existing progression-engine helpers when new scenarios require additional setup. Avoid introducing parallel database or monkeypatch scaffolding unless the existing fixtures cannot express the new flow.
- Reuse and extend existing pytest markers or fixture parameters when adding deterministic checks; do not fork the fixture graph or duplicate environment bootstrapping.
