## Parser and retry policy (authoritative)

- Every LLM answer must end with exactly one JSON object.
- `parse_json_safe` extracts the first JSON object and rejects answers with trailing text (raising "Trailing content detected after JSON object").
- If no JSON is found or schema validation fails:
  - The first call uses the standard `generate_with_continue` path, which is tagged with contextual markers such as `chat`, `followup_required`, `bloom_k1`, `struggles_present`, etc.
  - On failure we trigger a **single schema retry**. `extra_context` carries the marker `failed JSON schema`, and the retry call extends the `path_decisions`/`path_taken` chain with the suffix `schema_retry` (for example `chat > bloom_k1 > schema_retry`) that is persisted in both request telemetry and `llm_metrics`.
  - We stop after that retry. If the response is still invalid, the endpoint returns HTTP 502 and **does not** write a `chat_ops_log` entry.
- Additional fallbacks:
  - Language enforcement: if the model ignores the locale directive, we run a `language_retry` (`path_taken` for example `chat > bloom_k1 > language_retry`).
  - Continuations: lengthy responses can be extended with up to `continuation_n` steps (`path_taken` grows to `chat > bloom_k1 > continuation_1`, `chat > bloom_k1 > continuation_1 > continuation_2`, …) before a final JSON boundary is validated.
  - HTTP 400 from the model backend triggers a minimal-request fallback (no extra parameters) before we surface a 502 to the client. The successful minimal call is tagged with the same contextual markers; we do not append a special suffix.
- Storage and telemetry:
  - `chat_ops_log.raw_response`: full LLM text of the successful answer from the last successful attempt (including JSON and any schema-retry correction).
  - `chat_ops_log.response_json`: validated JSON payload (if available).
  - `llm_metrics.path_taken`: serialized retry/fallback chain using the contextual markers described above (`"chat > bloom_k1"`, `"chat > bloom_k1 > schema_retry"`, …). The same sequence is supplied as `path_decisions` on every LLM call so in-flight telemetry, request logging, and the database all align.
  - `llm_metrics.json_validated`: set to 1 after successful parsing, otherwise remains 0 (for example when we return a 502).

See `tests/test_chat_json_schema.py` for the exhaustive reference of schema retry flows.
