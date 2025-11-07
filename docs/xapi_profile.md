# AILearnBuddy xAPI profile

AILearnBuddy emits a constrained subset of xAPI statements to describe learner
activity. Statements are validated server-side before they are persisted or
forwarded to an external Learning Record Store (LRS). The profile currently
recognises the following verbs:

| Verb IRI | Display | Intended usage |
| --- | --- | --- |
| `http://adlnet.gov/expapi/verbs/answered` | answered | Learner submits an answer for a quiz item, activity, or checkpoint. |
| `http://adlnet.gov/expapi/verbs/evaluated` | evaluated | System evaluates a learner submission and records the score and supporting metadata. |
| `http://adlnet.gov/expapi/verbs/experienced` | experienced | Learner views a resource that requires teacher review or represents a notable learning event. |
| `http://adlnet.gov/expapi/verbs/initialized` | initialized | Learner session is created or an activity begins. |
| `http://adlnet.gov/expapi/verbs/mastered` | mastered | Learner achieves mastery for a Bloom level or skill. |
| `http://adlnet.gov/expapi/verbs/terminated` | terminated | Learner session or activity completes or exits. |

## Context extensions

Statements include a `context.extensions` object with a curated set of fields to
surface analytics and support governance checks:

- `bloom` (string): the Bloom level associated with the event (e.g. `K2`).
- `skill` (string): canonical skill identifier such as `math.algebra`.
- `confidence` (number 0-1): learner-reported or inferred confidence.
- `path` (string): pathway or content source indicator.
- `model_version` (string): LLM or scoring model version responsible for evaluation.
- `reason` (string): reason supplied for manual overrides or escalations.
- `session_type` (string): categorisation for the learning session (e.g. `tutoring`).
- `session_id` (string): identifier for the learning session.
- `subject` (string): high-level subject or domain identifier.
- `topic` (string): fine-grained topic or standard identifier for the activity.
- `decision` (string): decision outcome (e.g. `needs_regrade`).
- `source` (string): origin of the signal.
- `window_days` (integer): analytics window considered when the record was
  created.
- `metadata`, `summary`, `details` (object): structured payloads associated with
  sessions or events.

Additional keys are rejected during validation to avoid leaking arbitrary data
into downstream systems.

## Forwarding behaviour

Validated statements are stored in the local `xapi_statements` table and then
forwarded asynchronously to the configured LRS endpoint using exponential
backoff (up to three attempts). This ensures transient connectivity issues do
not block learner progress while still keeping the LRS in sync.
