import json

import pytest
import requests

from core import ollama_client
from core.ollama_client import (
    DEFAULT_BASE_URL,
    MODEL_NAME,
    build_messages,
    build_schema,
    check_model,
    generate_test_cases,
)
from core.scenarios import get_scenario

DISCOUNT = get_scenario("discount")
SHIPPING = get_scenario("shipping")


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text else (json.dumps(payload) if payload is not None else "")
        self.ok = status_code < 400
        self._json_error = payload is None and status_code == 200

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def chat_reply(content, **extra):
    payload = {"message": {"role": "assistant", "content": content}}
    payload.update(extra)
    return FakeResponse(200, payload)


def test_health_check_reports_ready_model(monkeypatch):
    tags = {"models": [{"name": "llama3:8b"}, {"name": MODEL_NAME}]}
    monkeypatch.setattr(ollama_client.requests, "get", lambda url, timeout: FakeResponse(200, tags))

    status = check_model()
    assert status.available is True
    assert status.state == "available"
    assert status.model == MODEL_NAME
    assert MODEL_NAME in status.installed_models
    assert status.installed_models == ("llama3:8b", MODEL_NAME)


def test_health_check_reports_missing_model_with_pull_hint(monkeypatch):
    tags = {"models": [{"name": "llama3:8b"}]}
    monkeypatch.setattr(ollama_client.requests, "get", lambda url, timeout: FakeResponse(200, tags))

    status = check_model()
    assert status.state == "model_missing"
    assert status.available is False
    assert f"ollama pull {MODEL_NAME}" in status.detail


def test_health_check_handles_connection_error(monkeypatch):
    def refuse(url, timeout):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(ollama_client.requests, "get", refuse)
    status = check_model()
    assert status.state == "unavailable"
    assert DEFAULT_BASE_URL in status.detail
    assert status.available is False


def test_health_check_handles_timeout(monkeypatch):
    def slow(url, timeout):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(ollama_client.requests, "get", slow)
    status = check_model(timeout=0.5)
    assert status.state == "unavailable"
    assert "0.5s" in status.detail


def test_health_check_handles_unreadable_payload(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests,
        "get",
        lambda url, timeout: FakeResponse(200, {"unexpected": True}),
    )
    status = check_model()
    assert status.state == "model_missing"
    assert status.installed_models == ()


@pytest.mark.parametrize("scenario", [DISCOUNT, SHIPPING], ids=lambda s: s.id)
def test_schema_is_scoped_to_the_scenario_arguments(scenario):
    schema = build_schema(scenario)
    cases = schema["properties"]["test_cases"]
    item = cases["items"]
    assert cases["minItems"] == 5
    assert cases["maxItems"] == 8
    assert item["properties"]["arguments"]["required"] == list(scenario.parameter_names)
    assert list(item["properties"]["arguments"]["properties"]) == list(scenario.parameter_names)
    assert item["additionalProperties"] is False
    assert item["required"] == ["arguments", "reason"]
    assert schema["additionalProperties"] is False


def test_prompt_includes_both_versions_and_forbids_predicted_outputs():
    messages = build_messages(DISCOUNT)
    assert [message["role"] for message in messages] == ["system", "user"]
    prompt = messages[1]["content"]
    assert DISCOUNT.contract in prompt
    assert DISCOUNT.original_source.strip() in prompt
    assert DISCOUNT.modified_source.strip() in prompt
    assert "Do not include expected outputs" in prompt
    assert "test INPUT DATA" in messages[0]["content"]


def test_generation_uses_real_chat_request_with_schema(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        captured["timeout"] = timeout
        return chat_reply('{"test_cases": []}', total_duration=123, eval_count=42)

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    result = generate_test_cases(DISCOUNT)

    assert result.ok is True
    assert result.raw_content == '{"test_cases": []}'
    assert result.used_schema is True
    assert result.total_duration_ns == 123
    assert captured["url"] == f"{DEFAULT_BASE_URL}/api/chat"
    assert captured["body"]["model"] == MODEL_NAME
    assert captured["body"]["stream"] is False
    assert captured["body"]["format"] == build_schema(DISCOUNT)
    assert captured["body"]["options"]["temperature"] <= 0.2
    assert captured["body"]["options"]["num_ctx"] == 4096


def test_generation_returns_timeout_category(monkeypatch):
    def slow(url, json=None, timeout=None):
        raise requests.Timeout("too slow")

    monkeypatch.setattr(ollama_client.requests, "post", slow)
    result = generate_test_cases(DISCOUNT, timeout=90)
    assert result.ok is False
    assert result.error_kind == "timeout"
    assert "90 seconds" in result.error_message


def test_generation_returns_offline_category(monkeypatch):
    def refuse(url, json=None, timeout=None):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(ollama_client.requests, "post", refuse)
    result = generate_test_cases(DISCOUNT)
    assert result.ok is False
    assert result.error_kind == "offline"
    assert DEFAULT_BASE_URL in result.error_message


def test_generation_detects_missing_model(monkeypatch):
    body = '{"error":"model \'qwen2.5-coder:7b\' not found, try pulling it first"}'
    monkeypatch.setattr(
        ollama_client.requests,
        "post",
        lambda url, json=None, timeout=None: FakeResponse(404, text=body),
    )
    result = generate_test_cases(DISCOUNT)
    assert result.ok is False
    assert result.error_kind == "model_missing"
    assert MODEL_NAME in result.error_message


def test_schema_rejection_falls_back_to_plain_json_and_says_so(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json["format"])
        if len(calls) == 1:
            return FakeResponse(400, text='{"error":"format schema unsupported"}')
        return chat_reply('{"test_cases": []}')

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    result = generate_test_cases(DISCOUNT)

    assert result.ok is True
    assert result.used_schema is False
    assert calls == [build_schema(DISCOUNT), "json"]
    assert "plain JSON mode" in result.notice


def test_server_error_is_returned_as_http_error(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests,
        "post",
        lambda url, json=None, timeout=None: FakeResponse(503, text="overloaded"),
    )
    result = generate_test_cases(DISCOUNT)
    assert result.ok is False
    assert result.error_kind == "http_error"
    assert "503" in result.error_message
    assert "overloaded" in result.detail


def test_empty_model_message_is_rejected(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests,
        "post",
        lambda url, json=None, timeout=None: chat_reply("   "),
    )
    result = generate_test_cases(DISCOUNT)
    assert result.ok is False
    assert result.error_kind == "empty_response"


def test_unparseable_http_payload_is_rejected(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests,
        "post",
        lambda url, json=None, timeout=None: FakeResponse(200, payload=None, text="<html>nope</html>"),
    )
    result = generate_test_cases(DISCOUNT)
    assert result.ok is False
    assert result.error_kind == "bad_response"


def test_no_hosted_fallback_is_configured():
    assert DEFAULT_BASE_URL.startswith("http://localhost:11434")
    assert MODEL_NAME == "qwen2.5-coder:7b"
