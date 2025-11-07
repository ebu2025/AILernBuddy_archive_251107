import app


def test_generate_with_continue_retries_for_english(monkeypatch):
    calls = []

    def fake_llm(messages, max_tokens, user_id=None, prompt_version=None, **kwargs):
        calls.append([m["content"] for m in messages])
        if len(calls) == 1:
            return "这是中文。"
        return "Here is the answer."

    monkeypatch.setattr(app, "_llm_call", fake_llm)

    result = app.generate_with_continue("System", "Question", 120)

    assert result == "Here is the answer."
    assert len(calls) == 2
    assert "Respond strictly in English" in calls[0][0]
    assert "You must provide the entire reply in English" in calls[1][0]


def test_generate_with_continue_returns_guard_message_when_non_english_persists(monkeypatch):
    calls = []

    def fake_llm(messages, max_tokens, user_id=None, prompt_version=None, **kwargs):
        calls.append([m["content"] for m in messages])
        return "继续写中文。"

    monkeypatch.setattr(app, "_llm_call", fake_llm)

    result = app.generate_with_continue("System", "Question", 120)

    assert result.startswith("I'm sorry, but I can only respond in English right now")
    assert len(calls) == 2
