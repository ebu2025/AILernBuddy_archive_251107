# Work Package Overview

This document summarises the current work packages (WP) linked to the assessment
and adaptive content roadmap. Items are scoped so engineering, data, and product
stakeholders can align on sequencing and dependencies.

## WP1 – Item Authoring & Metadata Hygiene
- Expand authoring tooling to capture mandatory metadata (domain, skill,
  Bloom level, answer schema) up front.
- Provide editors with validation hints sourced from `ItemBank.REQUIRED_FIELDS`
  so authors resolve issues before publishing.
- Output JSON compatible with `ItemBank` to guarantee downstream compatibility.

## WP2 – Retrieval Infrastructure Hardening
- Deliver ingestion/refresh tooling that **wraps the shipped helpers**:
  `rag.load_learning_materials`, `rag.split_documents`, `rag.build_vector_store`,
  and `rag.default_embedding_backend`. CLI entry points and orchestration
  scripts should import these helpers (or reuse the FastAPI bootstrap
  `_ensure_rag_ready`) so offline rebuilds share the same chunking, embedding,
  and persistence defaults as the running service.
- Extend configuration hooks—environment variables, dependency injection, or
  wiring in `_ensure_rag_ready`—that feed directly into
  `default_embedding_backend`/`build_vector_store`. Enhancements should expose
  persistence toggles (e.g., Chroma directories) and telemetry around
  `VectorStore.query` **without creating parallel embedding/vector-store
  implementations**.
- Update regression tests (`tests/test_rag_pipeline.py`, FastAPI endpoint
  suites, and any new CLI coverage) by exercising the existing helpers and
  endpoints. Future scenarios must extend these modules instead of duplicating
  ingestion or retrieval flows.

## WP3 – Item Bank Audits & Reporting
- Build CLI/reporting utilities that instantiate `ItemBank` and reuse its loader
  (`ItemBank._load`) for every audit run. This ensures the authoritative
  validation logic executes before any report is generated.
- Extend audits by composing additional checks on top of the in-memory
  `ItemBank.items` collection (e.g., Bloom coverage, metadata completeness)
  rather than parsing JSON separately.
- When new validation rules or quality metrics are needed, add them to
  `ItemBank` (or helpers it exposes) and exercise them through regression tests
  in `tests/test_item_bank.py` so the checks stay coupled to the existing API.
- Avoid duplicating parsing logic in standalone scripts; always call into the
  `ItemBank` API to keep audits, CLIs, and services aligned and to minimise
  maintenance overhead.

## WP4 – Language Strategy Registry Extensions
- Treat `_build_language_strategies` as the single registry for CEFR/HSK (and
  future) templates. New languages or variants should register themselves by
  contributing JSON files beneath `data/language/` or by adding lightweight
  configuration hooks—not by rebuilding loader code paths.
- Focus engineering effort on extensibility helpers around the registry:
  schema validators, configuration discovery utilities, and migration aids that
  plug into `_build_language_strategies` without forking its logic.
- Provide template authoring guidance and JSON examples that follow the current
  `levels -> {lexicon_modes, prompts}` structure so additional assets can plug
  in without modifying `TextGenerationProgressionEngine`.
- Expand automated checks (for example in
  `tests/test_text_generation_engine.py`) to enforce schema requirements as the
  registry grows, ensuring new contributors can extend strategies safely without
  duplicating loader logic.

## WP5 – Learner Journey Insights Automation
- (Reserved for future scope.)

## WP6 – Test Determinism & Fixture Stewardship
- Document the canonical fixtures and deterministic patterns in
  `docs/testing.md` (for example, `temp_db`, `_prepare_chat_environment`, and
  the progression engine score windows) so test authors have a single source of
  truth before adding new cases.
- When expanding coverage, extend the existing pytest markers and fixtures
  instead of recreating temporary database setups or monkeypatch harnesses.
  Capture any new parameters or helper methods directly on those fixtures.
- Add focused regression tests that cover clear behavioural gaps (e.g., new
  score thresholds or schema branches) using the documented patterns rather than
  cloning entire fixture graphs.
