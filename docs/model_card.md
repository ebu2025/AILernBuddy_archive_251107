# AILearnBuddy Model Card (MVP)

## Model details

* **Version:** 4.4.6 (FastAPI service orchestrating multiple instructional engines).
* **Developers:** AILearnBuddy open-source contributors.
* **Primary components:**
  * Domain Adaptive Orchestrator for knowledge graph traversal and Bloom-level tagging. 【F:app.py†L41-L52】【F:app.py†L60-L73】
  * Progression and ELO engines for mastery updates and skill calibration. 【F:app.py†L45-L53】【F:app.py†L108-L118】
  * Large-language-model integrations (via `_LLM_LOGGER` and `chat_ops_log`) for dialogue and instructional content generation. 【F:app.py†L91-L110】【F:db.py†L164-L171】【F:db.py†L3464-L3499】

## Intended use

The system generates adaptive learning journeys, assessments, and feedback for K-level through Bloom taxonomy-aligned topics. It is designed for pilot deployments where teachers supervise AI-generated recommendations. Supported scenarios include:

* Diagnostics and calibration quizzes (`/diagnose/start`, `/journey/diagnostic/start`). 【F:app.py†L1999-L2045】
* Adaptive learning path updates, including manual overrides for human control. 【F:app.py†L2159-L2187】【F:db.py†L2719-L2874】
* Chat-based tutoring sessions with audit logs for transparency. 【F:app.py†L144-L151】【F:db.py†L3430-L3499】

Any deployment should involve an educator or administrator reviewing analytics dashboards before acting on AI recommendations.

## Out-of-scope use cases

* High-stakes decisions (grading, certification, disciplinary action) without human verification.
* Sensitive domains requiring regulatory compliance beyond educational data (e.g., healthcare).
* Learners under strict data-protection regimes without institutional approval.

## Training data and sources

The orchestration logic relies on curated subject matter stored in SQLite (`item_bank`, `subjects`, `lessons`, etc.) and dynamic prompts persisted through `/prompts` endpoints. The system does not train new language models on learner data; instead, it logs prompts and responses for auditability. 【F:db.py†L152-L171】【F:db.py†L179-L234】【F:db.py†L3456-L3499】

## Evaluation

Automated evaluation harnesses `/eval/report` and `/eval/export` to measure response quality (accuracy, latency, JSON validity). These metrics are intended for offline monitoring and should be accompanied by teacher review. 【F:app.py†L1888-L1932】

## Ethical considerations and risks

* **Bias and fairness:** Content recommendations depend on seed items and manual inputs; any imbalance in the knowledge graph propagates to learners. Continuous review of `item_bank` and Bloom-level mappings is required.
* **Over-reliance on automation:** The system surfaces AI-suggested operations, but educators must validate them before execution. Pending operations remain reviewable in `chat_ops_log` until applied. 【F:db.py†L3430-L3499】
* **Data privacy:** Personal progress data persists until deletion. Operators must comply with local regulations and use the privacy endpoints for consent, export, and erasure. 【F:app.py†L1913-L1977】【F:db.py†L3358-L3650】

## Failure modes and mitigations

| Failure mode | Mitigation |
| --- | --- |
| Incorrect mastery updates leading to unsuitable recommendations | Teachers can apply manual overrides; learners can request recalibration via diagnostic sessions. 【F:app.py†L1999-L2045】【F:app.py†L2514-L2555】【F:db.py†L2880-L2928】 |
| Low K5/K6 coverage reduces higher-order practice opportunities | CLI validators and unit tests enforce minimum high-level Bloom coverage; failing ratios alert operators to rebalance content before release. 【F:scripts/validate_bloom_coverage.py†L129-L178】【F:tests/test_bloom_validator.py†L27-L66】 |
| LLM hallucinations in tutoring responses or hints | Logs (`chat_ops_log`) retain raw responses and applied operations so humans can review and correct misleading feedback, with hint activity surfaced to teacher analytics. 【F:db.py†L3430-L3499】【F:db.py†L2962-L3070】 |
| Data export/deletion failures due to missing user IDs | Endpoints validate `user_id` and raise HTTP 400/404 errors when records are incomplete, prompting operators to correct inputs. 【F:app.py†L1948-L1977】 |

## Human oversight

Educators interact with dedicated admin tools (e.g., `/teacher/learning-path/override`) that stamp overrides with `applied_by` identifiers and retain manual override histories. The teacher analytics dashboard highlights hint bursts, regression flags, and existing overrides, and lets staff submit annotated adjustments that persist in the learning path state. 【F:app.py†L2514-L2555】【F:db.py†L2880-L2928】【F:db.py†L2962-L3070】【F:static/ui.js†L380-L444】【F:static/admin.js†L560-L635】

## Maintenance and updates

* Database migrations add columns safely via `_add_column_if_missing`, preserving existing learner records. 【F:db.py†L52-L119】
* Logging uses FastAPI lifespan hooks to initialise databases and seed content, ensuring reproducible deployments. 【F:app.py†L27-L39】
* Operators should document any external LLM providers, model versions, or prompt templates used in production alongside this card.
