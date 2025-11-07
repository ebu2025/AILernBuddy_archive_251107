# Learning Journey Timeline Insights

This reference outlines the derived analytics exposed by `LearningJourneyTracker.get_timeline` so UI surfaces can render the
progress cards and nudges described in the analytics dashboards.

## Response Structure Overview

`get_timeline` returns a JSON object with the following top-level keys:

| Key | Description |
| --- | --- |
| `sessions` | Most recent sessions ordered by `started_at`. Each session now includes an `insights` block with derived metrics. |
| `loose_events` | Recent events that are not attached to a kept session. |
| `summary` | Aggregate counts plus roll-ups of the derived analytics. |
| `insights` | Timeline-wide analytics derived from all returned events. |
| `nudges` | Optional intervention recommendations such as inactivity alerts or failure streak prompts. |

## Session Insight Fields

Each session dictionary has an `insights` object containing:

| Field | Description |
| --- | --- |
| `duration_seconds` | Elapsed time from `started_at` to `ended_at`. If the session is still open, the latest event timestamp is used. |
| `event_count` | Number of events captured for the session. |
| `first_activity_at` / `last_activity_at` | ISO 8601 timestamps (UTC) for the earliest and most recent activity observed in the session. |
| `bloom` | Object capturing Bloom-level analytics: `levels_encountered`, `last_level`, and ordered `transitions` (`from`, `to`, `at`). |
| `streaks` | Session-scoped streak metrics with `current_success_streak`, `current_failure_streak`, `longest_success_streak`, and `total_failures`. |

### Event Outcomes

Outcome streaks infer success/failure using (in order):

1. Explicit booleans in `details.success`, `details.correct`, `details.passed`, or `details.completed`.
2. String or numeric outcomes (e.g., `details.outcome` equals "correct" or "failed").
3. Score heuristics (`score >= 0.7` is success, `score <= 0.3` is failure).

## Timeline-Level Insights

The top-level `insights` object summarises the consolidated timeline:

- `total_duration_seconds`: Sum of session durations returned in this timeline window.
- `last_activity_at`: Timestamp of the most recent activity across sessions and loose events.
- `bloom`: Overall progression with deduplicated `levels_encountered`, the latest observed level, and ordered `transitions`.
- `streaks`: Current and longest success streaks plus the cumulative failure count across the timeline window.

The `summary` object mirrors the counts required by existing dashboards while exposing the most critical derived indicators
(`total_duration_seconds`, `distinct_bloom_levels`, streak metrics, and `last_activity_at`).

## Nudges

`nudges` is a list of structured prompts generated while scanning the timeline:

- **Inactivity alerts** trigger when the learner has been inactive for more than 48 hours (`code: "inactivity"`).
- **Failure streak reminders** raise when three or more consecutive failures are detected (`code: "failure_streak"`).

Each nudge includes a `message`, `triggered_at` timestamp, and contextual metadata (e.g., `hours_inactive`). These cues align with
the intervention guidance in the learning journey best practices and can be surfaced in progress cards or notifications.
