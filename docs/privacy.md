# Privacy Note

AILearnBuddy is an adaptive learning prototype that follows data-minimisation and transparency principles even during pilot deployments. This note explains which learner and educator data the system stores, how long it is retained, and how individuals can exercise control.

## Data collected, purpose, and retention

| Data category | Where stored | Purpose | Retention |
| --- | --- | --- | --- |
| Account identifiers (`user_id`, optional email) | `users` table | Identify learners/educators and persist consent decisions. | Retained until a deletion request is received so accounts remain linkable to dependent records. |
| Consent records (`consented`, `consent_text`) | `user_consent` table | Track whether a person has agreed to data use. | Retained alongside the associated user account to evidence consent history until deleted. |
| Instructional content metadata (subjects, modules, lessons, activities) | `subjects`, `modules`, `lessons`, `activities` tables | Power domain models and Bloom-level pathways. | Retained until content is reauthored or the linked user account is removed. |
| Learning path state, manual overrides, and analytics | `learning_path_state`, `learning_path_events`, `teacher_analytics` tables | Generate personalised recommendations while allowing teacher oversight. | Retained until a learner deletion request clears the learner-scoped rows. |
| Mastery estimates and Bloom-level proficiency (`mastery`, `bloom_score`, `bloom_progress`, related history) | Corresponding tables | Maintain adaptive progression models. | Retained to inform adaptivity until the learner invokes deletion. |
| Assessment attempts and follow-ups (scores, rubric criteria, latency metrics) | `quiz_attempts`, `assessment_results`, `assessment_followups`, `llm_metrics` tables | Evaluate learner responses and track model performance. | Retained for longitudinal mastery tracking until the learner requests deletion. |
| Learning journey and chat interaction logs (operations, prompts, responses, applied actions) | `journey_log`, `chat_ops_log`, `pending_ops` tables | Provide auditability and replay of system decisions. | Retained for auditability until the learner deletion workflow purges the entries. |
| Learning events and xAPI statements (`event_type`, `score`, `context`) | `learning_events`, `xapi_statements` tables | Support interoperability analytics and teacher dashboards. | Retained until administrators action a deletion request for the learner. |
| Feedback on AI answers (`rating`, `comment`, `tags`) | `answer_feedback` table | Collect human judgements to improve future responses. | Retained until the originating learner or administrator requests deletion. |

Additional derived metrics (e.g., aggregated teacher analytics) are linked to the originating `user_id` for traceability. No biometric, financial, or precise geolocation data are collected.

### Lawful basis

Processing relies on explicit learner or guardian consent for analytics features and on the legitimate interests of educators delivering instruction within the pilot deployment.

## Retention and deletion

AILearnBuddy keeps learner records until they are explicitly deleted. Automated expiry is not yet implemented in this MVP release. Individuals or administrators can request deletion through the `/privacy/delete` endpoint, which removes records from all learner-scoped tables, including consent logs, cached analytics, and user accounts. 【F:app.py†L1969-L1977】【F:db.py†L3660-L3691】

## Data export and portability

Learners can obtain a structured export of their information via the `/privacy/export` endpoint. The export bundles profile, progress, assessment, journey, analytics, and consent artefacts with JSON decoding applied where relevant to preserve interpretability. 【F:app.py†L1948-L1963】【F:db.py†L3560-L3607】

| Endpoint | Method | Description | Authentication expectations |
| --- | --- | --- | --- |
| [`/privacy/export`](../app.py#L2178) | GET | Returns a JSON bundle of the learner’s stored records for portability. | Should be limited to the requesting learner or authorised administrator. |
| [`/privacy/delete`](../app.py#L2189) | DELETE | Initiates cascading deletion of learner-scoped records across analytics, mastery, journey, consent, and account tables. | Should be restricted to the learner or an administrator with verified authority. |

## Minimisation and access controls

* **Account provisioning and consent:** APIs call `db.ensure_user` to create placeholder learner records before persisting activity data. Consent is tracked via `/privacy/consent`, but flows must ensure approval is collected prior to continued processing. 【F:app.py†L1913-L1963】【F:db.py†L710-L717】
* **Role-informed operations:** Teacher interventions (e.g., manual path overrides) are stored with the `applied_by` identifier, ensuring auditable human oversight of automated recommendations. 【F:app.py†L2159-L2187】【F:db.py†L2719-L2874】
* **Operational logging:** `chat_ops_log` retains both automated and pending actions, enabling administrators to review and veto AI-suggested steps before they are applied.

All databases use SQLite in Write-Ahead Logging (WAL) mode and reside on infrastructure controlled by the deployment operator. Access should be restricted via operating-system permissions and network isolation as part of deployment hardening.

## User controls and support

1. **Consent:** POST `/privacy/consent` records acceptance or withdrawal along with the text presented at the time. 【F:app.py†L1913-L1946】【F:db.py†L3358-L3387】
2. **Export:** GET `/privacy/export?user_id=...` returns the data bundle described above.
3. **Deletion:** DELETE `/privacy/delete?user_id=...` erases learner-linked rows across analytics, mastery, journey, and consent tables.
4. **Human escalation:** Teachers can override or reverse automated recommendations through `/teacher/learning-path/override`, ensuring a human remains in control of progression-critical decisions. 【F:app.py†L2159-L2187】【F:db.py†L2719-L2874】

For production deployments, operators should pair these controls with secure authentication, encrypted storage, and periodic audits to satisfy UNESCO/OECD transparency expectations.
