import pytest

import db
import tutor


def test_microcheck_rubric_partial_credit_round_trip(temp_db):
    _ = temp_db
    user_id = "test-user"
    topic = "language"
    rubric = {"criteria": ["meaning"], "expected": "definition"}

    db.set_needs_assessment(
        user_id,
        topic,
        True,
        microcheck={
            "question": "Define the concept in one sentence.",
            "answer_key": "definition",
            "rubric": rubric,
        },
    )

    state = db.get_followup_state(user_id, topic)

    assert state is not None
    assert state["needs_assessment"] is True
    stored_rubric = state.get("microcheck_rubric")
    assert isinstance(stored_rubric, dict)
    assert stored_rubric.get("criteria") == ["meaning"]
    assert stored_rubric.get("keywords") == ["meaning", "definition"]

    reply = "It explains the meaning in our recent context."
    score = tutor.score_microcheck(reply, state.get("microcheck_answer_key"), stored_rubric)

    assert score == pytest.approx(0.5)

