import json

import pytest
from pydantic import ValidationError

from schemas import AssessmentResult, parse_json_safe


def _sample_payload() -> dict[str, object]:
    return {
        "user_id": "learner",
        "domain": "math",
        "item_id": "item-1",
        "bloom_level": "K3",
        "response": "Answer",
        "score": 0.82,
        "rubric_criteria": [
            {"id": "clarity", "score": 0.8},
            {"id": "accuracy", "score": 0.85},
        ],
        "model_version": "v-test",
        "prompt_version": "rubric.v1",
        "latency_ms": 1200,
        "tokens_in": 300,
        "tokens_out": 180,
    }


def test_parse_json_safe_rejects_trailing_payload():
    noisy_text = (
        "The model responded as follows:\n"
        "```json\n"
        f"{json.dumps(_sample_payload())}\n"
        "```\nThanks."
    )

    with pytest.raises(ValidationError):
        parse_json_safe(noisy_text, AssessmentResult)


def test_parse_json_safe_accepts_clean_json():
    payload = _sample_payload()

    result = parse_json_safe(json.dumps(payload), AssessmentResult)

    assert result.user_id == "learner"
    assert result.bloom_level == "K3"
    assert len(result.rubric_criteria) == 2
    assert result.rubric_criteria[0].id == "clarity"
    assert result.score == pytest.approx(0.82)
