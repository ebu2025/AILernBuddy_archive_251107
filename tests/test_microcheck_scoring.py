import pytest

import tutor


def test_score_microcheck_semantic_similarity_handles_synonyms():
    rubric = {"criteria": ["mentions exclusivity", "contrasts with parallel flow"]}
    reply = "It lets only a single branch continue while the other paths stay inactive."
    score = tutor.score_microcheck(
        reply,
        "Use it when exactly one path can proceed",
        rubric,
    )

    assert score >= 0.6


@pytest.mark.parametrize(
    "reply",
    [
        "The term captures the meaning we highlighted.",
        "It restates the meaning from our discussion.",
    ],
)
def test_score_microcheck_keyword_fallback_preserved(reply):
    rubric = {"criteria": ["meaning"], "expected": "definition"}
    score = tutor.score_microcheck(reply, "definition", rubric)
    assert score == pytest.approx(0.5)
