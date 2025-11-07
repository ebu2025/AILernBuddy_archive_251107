import asyncio
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if "httpx" not in sys.modules:
    stub = types.ModuleType("httpx")

    class _StubAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, *args, **kwargs):
            raise NotImplementedError("httpx stub does not support network calls")

        async def aclose(self):
            pass

    class _StubHTTPStatusError(Exception):
        def __init__(self, *args, response=None, **kwargs):
            super().__init__(*args)
            self.response = response

    class _StubTimeoutException(Exception):
        pass

    stub.AsyncClient = _StubAsyncClient
    stub.HTTPStatusError = _StubHTTPStatusError
    stub.TimeoutException = _StubTimeoutException
    sys.modules["httpx"] = stub

import app
import tutor


class _DummyResponse:
    def __init__(self, status_code, payload, requests_module):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)
        self._requests_module = requests_module

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise self._requests_module.HTTPError(response=self)


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class _DummyRequests:
    class HTTPError(Exception):
        def __init__(self, *args, response=None, **kwargs):
            super().__init__(*args if args else ("HTTP error",))
            self.response = response

    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if len(self.calls) == 1:
            return _DummyResponse(400, {"error": "invalid"}, self)
        return _DummyResponse(200, {"choices": [{"message": {"content": "Answer"}}]}, self)


class LlmFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._prev_client = getattr(app, "_http_client", None)
        self._prev_send_flag = app.SEND_MAX_TOKENS
        self._prev_requests = app.requests
        app.requests = _DummyRequests()
        app.SEND_MAX_TOKENS = True

    async def asyncTearDown(self):
        app._http_client = self._prev_client
        app.requests = self._prev_requests
        app.SEND_MAX_TOKENS = self._prev_send_flag

    async def test_fallback_retries_with_minimal_payload(self):
        messages = [{"role": "user", "content": "Hello"}]
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, app._llm_call, messages, 99)

        requests_stub = app.requests
        self.assertIsInstance(requests_stub, _DummyRequests)
        self.assertEqual(result, "Answer")
        self.assertEqual(len(requests_stub.calls), 2)

        first_call, second_call = requests_stub.calls
        self.assertIn("temperature", first_call["json"])
        self.assertEqual(
            second_call["json"],
            {"model": tutor.MODEL_ID, "messages": messages, "max_tokens": 99},
        )


class LlmFallbackUnitTests(unittest.TestCase):
    def setUp(self):
        self._prev_send_flag = app.SEND_MAX_TOKENS
        app.SEND_MAX_TOKENS = True

    def tearDown(self):
        app.SEND_MAX_TOKENS = self._prev_send_flag

    def test_fallback_retries_with_expected_payloads(self):
        messages = [{"role": "user", "content": "Hello"}]

        expected_timeout = app._safe_int("LLM_TIMEOUT", 1800)
        expected_base_payload = {
            "model": tutor.MODEL_ID,
            "messages": messages,
            **app._base_params(),
            "max_tokens": 99,
        }
        minimal_payload = {
            "model": tutor.MODEL_ID,
            "messages": messages,
            "max_tokens": 99,
        }

        calls = []

        def _fake_post(url, json=None, timeout=None):
            calls.append({"url": url, "json": json, "timeout": timeout})
            if len(calls) == 1:
                return _FakeResponse(400, {"error": "invalid"})
            return _FakeResponse(200, {"choices": [{"message": {"content": "Answer"}}]})

        with patch("app.requests.post", side_effect=_fake_post):
            result = app._llm_call(messages, max_tokens=99)

        self.assertEqual(result, "Answer")
        self.assertEqual(len(calls), 2)

        first_call, second_call = calls
        self.assertEqual(first_call["url"], tutor.GPT4ALL_URL)
        self.assertEqual(first_call["json"], expected_base_payload)
        self.assertEqual(first_call["timeout"], expected_timeout)

        self.assertEqual(second_call["url"], tutor.GPT4ALL_URL)
        self.assertEqual(second_call["json"], minimal_payload)
        self.assertEqual(second_call["timeout"], expected_timeout)


if __name__ == "__main__":
    unittest.main()
