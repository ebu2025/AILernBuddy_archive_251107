## AILearnBuddy — Copilot / AI agent instructions

Be concise and make edits that are small, well-tested, and minimally invasive. Below are the project-specific facts, patterns, and files an AI coding agent should know to be immediately productive.

1) Big picture (what this repo is)
- AILearnBuddy is an adaptive learning tutor (FastAPI) that combines: a progression engine, mastery/Bloom bookkeeping, an LLM-driven copilot for lesson plans, and optional RAG retrieval. Key runtime entrypoint: `app.py`.
- Core subsystems and where to look:
  - HTTP API + wiring: `app.py` (middleware, singletons, endpoint overview). See endpoints like `/chat`, `/ask`, `/copilot/plan`, `/copilot/moderate` in README and `app.py`.
  - Engines: `engines/` (e.g., `progression.py`, `elo.py`, `competency.py`, `domain_adapter.py`) — these are instantiated as singletons inside `app.py` (e.g., `PROGRESSION_ENGINE`, `ELO_ENGINE`). Treat them as the canonical business logic layer.
  - Tutor prompts & safety: `tutor.py` (prompt versions live as `PROMPT_VERSION`, `PROMPT_VARIANT`). Follow existing prompt patterns when adding new prompts.
  - Knowledge graph / path planning: `knowledge_graph.py`, `graph_path_planner.py`, `learning_path.py` — used for content recommendations and learning journeys.
  - Persistence & telemetry: `db.py`, `xapi.py`, and `llm_metrics` table. Tests and scripts expect SQLite local storage by default.

2) Developer workflows & commands (how to run and test)
- Local dev (POSIX/Windows differences): use `run_server.sh` (Unix) or `run_server.bat` (Windows). Both create/activate `.venv`, install `requirements.txt`, set recommended env vars and start `uvicorn app:app --reload --port 8000`.
- Required env vars you may need to set when testing LLM features:
  - `GPT4ALL_URL` (e.g. `http://localhost:4891/v1/chat/completions`)
  - `MODEL_ID` (sample: `DeepSeek-R1-Distill-Qwen-14B`)  
  - `ENGINE_MODE` (e.g., `elo`, `hybrid`) and LLM tuning vars (`LLM_TIMEOUT`, `SEND_MAX_TOKENS`, `MAX_TOKENS`)
- Tests: run `pytest` for the suite. For Bloom coverage, run `python scripts/bloom_validate.py --items items.json --output tests/baselines/bloom_coverage.json`, then `pytest tests/test_bloom_validator.py` to enforce the baseline.

3) Project-specific conventions & patterns
- Singletons in `app.py`: many subsystems are created once and shared (e.g., `LEARNING_PATH_MANAGER`, `DOMAIN_ORCHESTRATOR`). When changing behavior, prefer adding configuration hooks or small adapters rather than re-instantiating these in multiple places.
- Token auth is lightweight and in-memory by default (`TOKENS` dict); protected endpoints are defined in `_PROTECTED_PATHS`. When adding endpoints that must be protected, add them there or use the existing middleware.
- Assessment cascade: the tutor writes structured `AssessmentResult` JSON where possible (see README section "Always-Learning Cascade"). New LLM flows must emit structured JSON or implement the self-check -> micro-assessment fallback.
- Prompt versioning: use `tutor.PROMPT_VERSION` and `PROMPT_VARIANT` to avoid unexpected prompt drift. Bump versions consciously and update tests/baselines if outputs change.

4) Integration points & external dependencies
- LLMs / RAG: the code integrates with local GPT4All or remote LLM endpoints (`GPT4ALL_URL`) and an optional RAG vector store (corpus: `docs/rag_corpus.json`, RAG control via `RAG_CORPUS_PATH` env). If you change RAG loading, check `_RAG_INITIALIZED` and `_ensure_rag_ready()` in `app.py`.
- xAPI / LRS: external LRS forwarding is controlled with `LRS_URL` and `LRS_AUTH` (see README). Without those env vars, statements are stored locally.
- DB: repo uses `db.py` (SQLite) by default. Tests read/write local DB rows — avoid destructive schema changes without migrations and test updates.

5) Tests & important baseline checks
- Bloom validator: `scripts/bloom_validate.py` and `tests/test_bloom_validator.py` are the canonical gate for `items.json` changes. A PR changing items MUST include an updated baseline and pass the validator locally.
- Capability panel & eval endpoints: `POST /eval/run` stores probe results in `eval_reports`; tests rely on returned `run_id` structure — be conservative changing formats.

6) Files worth opening when working in new areas
- `app.py` — wiring, auth, singletons, protected paths
- `tutor.py` — prompts, LLM interactions, cascade logic
- `engines/` — progression, ELO, competency definitions, graph planner
- `db.py`, `xapi.py` — persistence and telemetry
- `scripts/bloom_validate.py`, `items.json`, `tests/baselines/` — bloom coverage workflow
- `docs/learning_journey_best_practices.md` and other docs under `docs/` for architecture rationale

7) Safe edit rules for AI agents (do these automatically)
- Keep changes small and focused. Prefer adding small helper functions or flags over large refactors.
- When modifying prompts, increment `PROMPT_VERSION` and add/adjust tests that assert the structured outputs (AssessmentResult JSON) or the Bloom baseline as needed.
- When touching persistence schemas, run the test suite and the Bloom validator locally; update baseline files in `tests/baselines/` only with a clear PR description.

If anything here is unclear or you'd like me to expand a section (CI, secret management, or detailed engine contracts), tell me which area to expand and I will iterate.
