# Evaluation Plan

## Purpose
This plan defines how AILearnBuddy evaluates learning outcomes for adaptive journeys.
It documents the instruments, sample requirements, metrics, and analysis workflow so
that evaluators can reproduce reports served by the `/eval/report` endpoint.

## Instruments
- **Pre-instruction instrument**: standardized diagnostic aligned to the selected topic.
  - Recorded via `POST /eval/pretest` with `instrument_id`, `instrument_version`, and
    optional metadata to capture form variants.
  - Must contain at least 10 scored items and report `score` and `max_score`.
- **Post-instruction instrument**: mirror or parallel form of the pretest.
  - Recorded via `POST /eval/posttest` with the same instrumentation fields to link
    attempts into comparable pre/post pairs.
  - Uses identical scoring rubric; alternative forms must be equated prior to upload.

## Sample Size Targets
- Minimum **30 matched pre/post pairs** per analytic cell (topic × strategy) before
  reporting normalized gains. Below this threshold, `/eval/report` still lists raw
  pairs but flags the cell for cautionary interpretation.
- Aim for **at least 100 matched pairs** overall for program-level summaries to reduce
  confidence interval widths. The `/eval/report` summary exposes `pair_count` and
  `valid_gain_count` so analysts can monitor achieved samples.

## Metrics (Aligned with `/eval/report`)
- **Δ (mean delta)**: average of `post_normalized - pre_normalized` across matched
  pairs. Available as `overall.mean_delta` and in the strategy/topic breakdown data.
- **Normalized gain (g)**: computed per pair as `(post - pre) / (1 - pre)` where
  scores are normalized to `[0, 1]`. Reported as `pairs[].normalized_gain` with
  aggregate `overall.average_normalized_gain` and a 95% confidence interval from
  `overall.gain_confidence_interval`.
- **Pre/post normalized means**: `overall.mean_pre` and `overall.mean_post` provide
  baseline and outcome anchors for interpreting Δ and g.

## Analysis Steps
1. **Collect attempts**: ensure learners complete both pre and post instruments with
   consistent `learner_id`, `topic`, and `strategy` so the database can pair them via
   `db.record_pretest_attempt` and `db.record_posttest_attempt`.
2. **Normalize scores**: the reporting pipeline rescales raw scores to the `[0, 1]`
   interval (fields `pre_normalized` and `post_normalized`). Analysts should verify
   that the underlying score ranges are consistent across instruments before upload.
3. **Generate pairs**: query `GET /eval/report` to retrieve `pairs` containing
   pre/post metrics, delta, and normalized gain for each learner-topic-strategy trio.
4. **Aggregate**: use the returned `summary` to review topic or strategy level means.
   The endpoint mirrors `db.summarize_normalized_gains`, ensuring documentation and
   dashboards share identical aggregation logic.
5. **Interpret**: focus on cells meeting the sample thresholds and interpret Δ and g
   jointly—Δ captures absolute improvement while g contextualizes improvement relative
   to headroom. Use the provided confidence interval to communicate uncertainty.
6. **Report**: include pair counts, Δ, g, and confidence intervals in stakeholder
   summaries. When exporting underlying records for audit, use `GET /eval/export` to
   obtain CSV-formatted probe metrics.

## Quality Checks
- Confirm no learner has multiple unmatched attempts; resolve duplicates before
  analysis.
- Spot-check instruments to ensure pre/post alignment and that metadata accurately
  reflects the administered form.
- Monitor latency and JSON validity metrics from `/eval/run` when validating new
  automated evaluation probes.
