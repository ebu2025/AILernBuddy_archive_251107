import tutor


def test_generate_microcheck_bloom_specific_question():
    low_level = tutor.generate_microcheck("BPMN gateways", bloom_level="K1")
    high_level = tutor.generate_microcheck(
        "BPMN gateways",
        bloom_level="K4",
        recent_answer="The diagram already contrasts parallel and exclusive gateways.",
    )

    assert "exclusive" in low_level["question"].lower()
    assert "risk" in high_level["question"].lower()
    assert low_level["question"] != high_level["question"]
    assert low_level["rubric"] != high_level["rubric"]


def test_generate_microcheck_tracks_recent_focus():
    microcheck = tutor.generate_microcheck(
        "language practice",
        bloom_level="K3",
        recent_answer="I struggled with translation accuracy in the last step.",
    )

    metadata = microcheck.get("metadata", {})
    assert metadata.get("focus") == "translation accuracy"


def test_generate_microcheck_includes_long_term_focus_metadata():
    microcheck = tutor.generate_microcheck(
        "math practice",
        bloom_level="K2",
        learning_focus="Lineare Gleichungen automatisiert lösen",
        learning_snapshot={"current_level": "K2", "progress_by_level": {"K2": 0.58}},
    )

    metadata = microcheck.get("metadata", {})
    assert metadata.get("long_term_focus") == "Lineare Gleichungen automatisiert lösen"
    snapshot = metadata.get("learning_snapshot")
    assert isinstance(snapshot, dict)
    assert snapshot.get("current_level") == "K2"
    assert "Langfristiges Ziel" in microcheck.get("hint", "")
    assert "Langfristiges Ziel" in microcheck.get("question", "")
