# AGENT — Minimal contributor guide for automated agents

This short guide shows how to perform small, safe edits and run a tiny local harness to smoke-check the repository.

Example edit workflow (minimal, safe):

1. Create a small branch for your change (feature or fix).
2. Run the unit tests most relevant to your change. Examples:
   - UI / prompts: `pytest tests/test_chat_json_schema.py::test_chat_json_schema -q`
   - Bloom/items: run the validator (see below).
3. If you modify prompts in `tutor.py` or add new prompt files, increment `tutor.PROMPT_VERSION` and add/adjust tests that assert structured JSON outputs.
4. If you change `items.json`, update the Bloom baseline:
   - `python scripts/bloom_validate.py --items items.json --output tests/baselines/bloom_coverage.json`
   - `pytest tests/test_bloom_validator.py -q`
5. Run the lightweight harness to ensure imports and HTTP routes are intact (example commands below).
6. Commit small, focused changes with clear commit messages and include tests where feasible.

PR review checklist (automatable):
- Does the change touch `items.json`? If yes, ensure the Bloom baseline is updated.
- Does the change alter prompts or output JSON shapes? If yes, bump `PROMPT_VERSION` and update affected tests.
- Did you run `pytest` (targeted tests at minimum)?
- Are any singletons in `app.py` being re-instantiated? Prefer configuration hooks or flags instead of duplicating singletons.

Tiny local test harness

The harness is a safe, fast smoke-check that imports the app and prints a summary of registered routes and the tutor prompt version.

Run the harness (PowerShell):

```powershell
# Activate venv first if needed
.venv\Scripts\Activate.ps1
python scripts\agent_test_harness.py
```

Run the harness (POSIX):

```bash
source .venv/bin/activate
python3 scripts/agent_test_harness.py
```

What the harness checks:
- `app` imports successfully.
- `app.title` and route paths are listed (quick sanity on API surface).
- `tutor.PROMPT_VERSION` is present (useful when changing prompts).

If the harness fails on import, check for missing dependencies in `requirements.txt` or environment variable requirements like `GPT4ALL_URL` / `MODEL_ID` if touching LLM/RAG code.

Next steps you can ask me to do:
- Expand this into a longer `CONTRIBUTING.md`.
- Add a GitHub Action that runs the harness + a targeted pytest subset for PRs.

---
Files created/used:
- `.github/AGENT.md` (this file)
- `scripts/agent_test_harness.py` (small harness)
