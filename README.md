# Local Tutor (GPT4All + DeepSeek) – Fully Operational

## Start
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Ensure GPT4All is running locally (http://localhost:4891) with the DeepSeek-R1-Distill-Qwen-14B model
$env:MODEL_ID="DeepSeek-R1-Distill-Qwen-14B"
$env:GPT4ALL_URL="http://localhost:4891/v1/chat/completions"
$env:ENGINE_MODE="hybrid"

uvicorn app:app --reload --port 8000
# -> http://127.0.0.1:8000/
```

## Bloom validator workflow

- `python scripts/bloom_validate.py --items items.json --output tests/baselines/bloom_coverage.json`
  generates the per-domain Bloom coverage report. Run this whenever you add or
  significantly edit items so the baseline stays in sync.
- `pytest tests/test_bloom_validator.py` enforces a minimum 30% share of K5–K6
  items per domain and fails if the report drifts from the saved baseline. CI
  runs this test automatically—update the baseline only when the new coverage is
  intentional.
- Pull requests that modify `items.json` must include an updated coverage
  baseline and pass the validator before merging.

## Endpoint Overview
- `POST /auth/register` | `POST /auth/login`
- `POST /chat` (soft operations, LLM answer with validated database suggestions)
- `POST /ask` (practice activities, grading, **engine** updates θ)
- `POST /diagnose/start` (start with three diagnostic items)
- `POST /journey/session/start` | `POST /journey/session/end` (open/close learning journey sessions – `session/end` now requires the explicit user ID and refuses foreign sessions)
- `POST /journey/event` (log fine-grained learning events)
- `GET /journey/timeline` (aggregate sessions + events for dashboards)
- `GET /teacher/analytics` (flagged cohorts with hint/confidence signals, learning path timelines, Bloom history, and aggregated learner feedback)
- `POST /bloom/score` (rubric score → Bloom level)
- `GET /profile` (open learner model with mastery and Bloom status)
- `POST /copilot/plan` (teacher co-design lesson plan with Bloom alignment + provenance)
- `POST /copilot/moderate` (teacher reviews AI artefacts, logs decision + rationale)
- `POST /eval/run` | `GET /eval/export` (capability panel run + CSV export of probe metrics)
- `GET /db/*` (mastery, prompts, journey, items, Bloom)
- `GET /privacy/export` | `DELETE /privacy/delete` (GDPR-style data export & deletion)

## Authentication

- Call `POST /auth/register` followed by `POST /auth/login` to receive a bearer token.
- Include `Authorization: Bearer <token>` (or `X-Token: <token>`) when invoking protected endpoints.
- `/chat`, `GET /privacy/export`, and `DELETE /privacy/delete` now reject requests that omit the token or present an unknown token.

## Always-Learning Cascade

Every tutor turn now passes through a four-stage cascade so that each interaction yields a learning signal:

1. **Direct JSON.** The assistant attempts to emit an `AssessmentResult` block with `confidence` and `source`. Valid JSON is accepted immediately.
2. **Self-check regrade.** If the JSON is missing or invalid, a lightweight regrade prompt is issued. Self-check verification reduces hallucinations by forcing the model to re-evaluate its own answer before we trust the score.
3. **Micro-assessment.** When both direct and self-check paths fail, a one-line micro check is queued for the next learner message and graded heuristically (confidence 0.5, source `heuristic`).
4. **Pending flag.** If we still have no structured evidence, a `PendingOp` row is recorded, an xAPI `experienced` statement is emitted, and the next turn is forced through the micro-check path.

The cascade writes provenance (`source`, `confidence`) into `assessment_results`, into the confidence-weighted progression engine, and onto `/profile` so lecturers can inspect the audit trail.

## Open Learner Model & Profile

- `GET /profile` exposes θ, `p_know` (logistic transform of θ), Bloom levels, the five most recent assessments (with confidence + source), the spaced-repetition due list (SM-2 style decay), and `feedback_summary` aggregating learner reactions (`answer_feedback`) so coaches can balance sentiment with evidence. Each skill entry includes the latest evidence ID so lecturers can trace decisions.
- `/teacher/analytics` returns the stuck learner snapshot enriched with `learning_path_events` (chronological for plotting), `latest_path_event` (reason + evidence summary), historical Bloom transitions, and `feedback_summary` roll-ups per learner-topic pair for quick drilling into support needs.
- Bloom transitions remain in `bloom_progress`, θ values in `mastery`, and both surfaces stay available via the privacy export and `/db/*` endpoints.

## Privacy & Export

- `GET /privacy/export?user_id=<id>` returns a JSON bundle of all relevant rows (`users`, `mastery`, `bloom_progress`, `learning_events`, `journey_log`, `xapi_statements`, `assessment_results`, `llm_metrics`, and more).
- `DELETE /privacy/delete?user_id=<id>` removes those rows in a safe order (events → progress → user entry).
- `.env.example` lists optional settings for connecting to an external LRS (`LRS_URL`, `LRS_AUTH`). Without those variables set, statements are stored locally in `xapi_statements` only.

## Telemetry & KPIs

LLM calls are recorded in `llm_metrics`. Example SQLite queries:

- **Median prompt-to-first-token (ms)**
  ```sql
  SELECT latency_ms
  FROM llm_metrics
  ORDER BY latency_ms
  LIMIT 1 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM llm_metrics);
  ```
- **Tokens per second**
  ```sql
  SELECT AVG(tokens_out * 1000.0 / NULLIF(latency_ms, 0)) AS tokens_per_sec
  FROM llm_metrics
  WHERE tokens_out IS NOT NULL;
  ```
- **p95 end-to-end latency (ms)**
  ```sql
  SELECT latency_ms
  FROM llm_metrics
  ORDER BY latency_ms
  LIMIT 1 OFFSET CAST(0.95 * (COUNT(*) - 1) AS INTEGER);
  ```

### Confidence-weighted Knowledge Tracing

- `engines/progression.ProgressionEngine` computes weighted averages where the assessment confidence acts as the weight. Promotion requires a weighted average ≥0.8 and total weight ≥2.0; two weighted fails trigger a demotion even if older history was strong.
- A TD-BKT (time-decayed Bayesian Knowledge Tracing) hook can be enabled via `ENABLE_TD_BKT=true`. Parameters (guess/slip/learn) stay explicit to keep the knowledge model interpretable for lecturers.

### xAPI Evidence Log

- `xapi.emit` records every path: `answered` (practice attempts), `evaluated` (assessment results), `experienced` (pending rechecks), and `initialized`/`terminated` for journey sessions.
- When `LRS_URL`/`LRS_AUTH` are set, statements are forwarded with `X-Experience-API-Version: 1.0.3`, matching ADL’s compliance guidance. Without an external LRS the SQLite log remains available for audit.

### Teacher Copilot & Moderation

- `POST /copilot/plan` drafts Bloom-tagged lesson plans with provenance, objectives, culture-sensitivity flags, and teacher notes. Plans are stored in `copilot_plans` with alignment metadata.
- `POST /copilot/moderate` captures moderator decisions, rationale, and flags; records live in `copilot_moderation` and are surfaced alongside the plan.
- Culture checks (`tutor.culture_sensitivity_check`) flag idioms/slang/taboo phrases so teachers see fairness signals before release.

### Capability Evaluation Panel

- `POST /eval/run` executes a fixed probe set across reasoning, knowledge QA, BPMN, and language. Results (accuracy, JSON validity, latency) are stored in `eval_reports` under a generated `run_id` and returned alongside a summary.
- `GET /eval/export` streams the results as CSV for analysis or thesis plots.
- Offline environments can keep `EVAL_USE_LLM=false`; the panel still records deterministic probes for regression tracking.

### Service Level Objectives (weekly targets)

- **Structured evidence:** Invalid assessment JSON after self-check repair < **1%** (`source` = `pending` entries in `assessment_results`).
- **Latency:** Median prompt-to-first-token ≤ **900 ms**, p95 ≤ **3000 ms** (query `llm_metrics`).
- **Capability panel:** Maintain ≥ **0.8** accuracy on reasoning and BPMN probes; disclose if RAG or self-check was used via the returned `used_llm` flag.
- **Learning outcomes:** Track rolling normalized score and Bloom transitions via `/profile` (fields `recent_attempts` and `spaced_due`). Teacher acceptance rate of copilot plans is observable via `copilot_moderation` decisions.
- **Fairness:** Monitor gap in accuracy/latency by locale (culture flags) and moderation flag rates before release.

## Additional Documentation

- [Learning Journey & History Modeling Best Practices](docs/learning_journey_best_practices.md): overview of data modelling, architecture, and governance for learning histories and adaptive learning paths in AILearnBuddy.
