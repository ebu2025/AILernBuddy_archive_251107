import math  # Required for math.isclose assertions

from process_models.process_mining import (
    calculate_cycle_times,
    discover_variants,
    generate_process_diagnostics,
    identify_bottlenecks,
    parse_event_log,
)


RAW_EVENTS = [
    {
        "case_id": "learner-2",
        "activity": "Visualization Lab",
        "timestamp": "2024-02-02T12:45:00",
        "skill_area": "data_fluency",
        "media_channel": "diagram_tools",
    },
    {
        "case_id": "learner-1",
        "activity": "Capstone",
        "timestamp": "2024-02-01T14:00:00",
        "skill_area": "foundational",
        "media_channel": "simulation",
    },
    {
        "case_id": "learner-1",
        "activity": "Intake",
        "timestamp": "2024-02-01T09:00:00",
        "skill_area": "foundational",
        "media_channel": "diagram_tools",
    },
    {
        "case_id": "learner-1",
        "activity": "Diagnostic",
        "timestamp": "2024-02-01T10:00:00",
        "skill_area": "foundational",
        "media_channel": "diagram_tools",
    },
    {
        "case_id": "learner-2",
        "activity": "Capstone",
        "timestamp": "2024-02-02T15:30:00",
        "skill_area": "data_fluency",
        "media_channel": "audio",
    },
    {
        "case_id": "learner-3",
        "activity": "Diagnostic",
        "timestamp": "2024-02-03T10:30:00",
        "skill_area": "data_fluency",
        "media_channel": "audio",
    },
    {
        "case_id": "learner-3",
        "activity": "Coaching",
        "timestamp": "2024-02-03T12:00:00",
        "skill_area": "data_fluency",
        "media_channel": "audio",
    },
    {
        "case_id": "learner-2",
        "activity": "Diagnostic",
        "timestamp": "2024-02-02T10:15:00",
        "skill_area": "data_fluency",
        "media_channel": "diagram_tools",
    },
    {
        "case_id": "learner-2",
        "activity": "Intake",
        "timestamp": "2024-02-02T09:30:00",
        "skill_area": "data_fluency",
        "media_channel": "diagram_tools",
    },
    {
        "case_id": "learner-1",
        "activity": "Workshop",
        "timestamp": "2024-02-01T12:00:00",
        "skill_area": "foundational",
        "media_channel": "simulation",
    },
    {
        "case_id": "learner-3",
        "activity": "Intake",
        "timestamp": "2024-02-03T09:00:00",
        "skill_area": "data_fluency",
        "media_channel": "diagram_tools",
    },
]


def test_parse_event_log_orders_by_case_and_time():
    events = parse_event_log(RAW_EVENTS)
    ordered = [(evt.case_id, evt.activity) for evt in events]
    assert ordered[:3] == [
        ("learner-1", "Intake"),
        ("learner-1", "Diagnostic"),
        ("learner-1", "Workshop"),
    ]
    assert ordered[-1] == ("learner-3", "Coaching")


def test_calculate_cycle_times_returns_hours():
    events = parse_event_log(RAW_EVENTS)
    cycle_times = calculate_cycle_times(events)
    assert math.isclose(cycle_times["learner-1"], 5.0)
    assert math.isclose(cycle_times["learner-2"], 6.0)
    assert math.isclose(cycle_times["learner-3"], 3.0)


def test_discover_variants_counts_sequences():
    events = parse_event_log(RAW_EVENTS)
    variants = discover_variants(events)
    assert variants["Intake > Diagnostic > Workshop > Capstone"] == 1
    assert variants["Intake > Diagnostic > Visualization Lab > Capstone"] == 1
    assert variants["Intake > Diagnostic > Coaching"] == 1


def test_identify_bottlenecks_ranks_longest_waits():
    events = parse_event_log(RAW_EVENTS)
    bottlenecks = identify_bottlenecks(events, top_n=2)
    assert bottlenecks[0][0] == "Visualization Lab"
    assert bottlenecks[0][1] > bottlenecks[1][1]


def test_generate_process_diagnostics_summarises_metrics():
    events = parse_event_log(RAW_EVENTS)
    diagnostics = generate_process_diagnostics(events, end_activity="Capstone")

    assert diagnostics["cases"] == 3
    assert diagnostics["variants"][0]["variant"] == [
        "Intake",
        "Diagnostic",
        "Workshop",
        "Capstone",
    ]
    assert diagnostics["dropouts"]["cases"] == ["learner-3"]
    assert diagnostics["skill_areas"]["foundational"]["mean_cycle_hours"] == 5.0
    assert diagnostics["skill_areas"]["data_fluency"]["cases"] == 2.0
    assert diagnostics["skill_areas"]["data_fluency"]["media_channels"]["audio"] == 3
    assert diagnostics["media_channels"]["foundational"]["simulation"] == 2
