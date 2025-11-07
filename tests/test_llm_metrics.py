import app
import db


def test_llm_call_records_metrics(monkeypatch, temp_db):
    db.init()

    responses = {
        "choices": [
            {"message": {"content": "Answer"}}
        ],
        "usage": {"prompt_tokens": 200, "completion_tokens": 120},
    }

    class _Resp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    def fake_post(url, json=None, timeout=None):
        return _Resp(responses)

    monkeypatch.setattr(app.requests, "post", fake_post)

    result = app._llm_call(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=128,
        user_id="alice",
        prompt_version=app.CHAT_PROMPT_VERSION,
        prompt_variant=app.CHAT_PROMPT_VARIANT,
    )

    assert result == "Answer"
    rows = db._query(
        "SELECT user_id, model_id, prompt_version, prompt_variant, latency_ms, tokens_in, tokens_out FROM llm_metrics"
    )
    assert len(rows) == 1
    entry = dict(rows[0])
    assert entry["user_id"] == "alice"
    assert entry["model_id"] == app.tutor.MODEL_ID
    assert entry["prompt_version"] == app.CHAT_PROMPT_VERSION
    assert entry["prompt_variant"] == app.CHAT_PROMPT_VARIANT
    assert entry["latency_ms"] >= 0
    assert entry["tokens_in"] == 200
    assert entry["tokens_out"] == 120
