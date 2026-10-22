"""Headless checks of the real Streamlit page's states.

Ollama is faked at the client boundary, so no server is needed. The pipeline,
validation, and rendering still run for real.
"""

import streamlit as st
from streamlit.testing.v1 import AppTest

from core import ollama_client
from core.ollama_client import MODEL_NAME, GenerationResult, ModelStatus

READY = ModelStatus("available", MODEL_NAME, (MODEL_NAME,), detail="ready")
OFFLINE = ModelStatus(
    "unavailable", MODEL_NAME, detail="Could not reach Ollama at http://localhost:11434."
)


def markdown_text(at: AppTest) -> str:
    return "\n".join(str(element.value) for element in at.markdown)


def run_app(monkeypatch, status, generation=None):
    monkeypatch.setattr(ollama_client, "check_model", lambda *a, **k: status)
    if generation is not None:
        monkeypatch.setattr(ollama_client, "generate_test_cases", lambda *a, **k: generation)
    st.cache_data.clear()
    return AppTest.from_file("app.py", default_timeout=20).run()


def test_page_renders_both_trusted_versions_and_the_real_status(monkeypatch):
    at = run_app(monkeypatch, READY)
    assert not at.exception
    text = markdown_text(at)
    assert "Find the break before it ships." in text
    assert "Local / Qwen2.5-Coder 7B · Ready" in text
    assert "def calculate_discount(price, discount):" in text
    assert "price - (price * discount / 100)" in text
    assert "No test run yet." in text
    assert "Ollama unavailable" not in text


def test_offline_state_is_actionable_and_shows_no_results(monkeypatch):
    at = run_app(monkeypatch, OFFLINE)
    assert not at.exception
    text = markdown_text(at)
    assert "Ollama / offline" in text
    assert "Ollama unavailable" in text
    assert "No hosted model will be substituted." in text
    assert "No test run yet." in text
    assert "Observed behaviour" not in text


def test_model_missing_state_names_the_pull_command(monkeypatch):
    status = ModelStatus("model_missing", MODEL_NAME, ("llama3:8b",), detail="not installed")
    at = run_app(monkeypatch, status)
    assert not at.exception
    text = markdown_text(at)
    assert "Ollama / model missing" in text
    assert "ollama pull qwen2.5-coder:7b" in text


def test_generation_transport_failure_is_shown_without_stale_results(monkeypatch):
    failure = GenerationResult(
        ok=False,
        error_kind="offline",
        error_message="Ollama is not reachable at http://localhost:11434.",
        detail="Start Ollama and confirm the model is installed.",
        model=MODEL_NAME,
        duration_seconds=0.1,
    )
    at = run_app(monkeypatch, READY, generation=failure)
    at.button[0].click().run()
    assert not at.exception
    text = markdown_text(at)
    assert at.button[0].label == "Generate & run tests"
    assert "Ollama is offline" in text
    assert "Observed behaviour" not in text
    assert "No test run yet." not in text


def test_invalid_json_reply_is_reported_as_ai_response_invalid(monkeypatch):
    generation = GenerationResult(
        ok=True,
        raw_content="Sure! Here is a test plan, not JSON.",
        model=MODEL_NAME,
        duration_seconds=2.2,
        used_schema=True,
    )
    at = run_app(monkeypatch, READY, generation=generation)
    at.button[0].click().run()
    assert not at.exception
    text = markdown_text(at)
    assert "AI response invalid" in text
    assert "not valid JSON" in text
    assert "Observed behaviour" not in text


def test_zero_usable_cases_reports_each_rejection(monkeypatch):
    generation = GenerationResult(
        ok=True,
        raw_content=(
            '{"test_cases": [{"arguments": {"price": true, "discount": 10}, '
            '"reason": "boolean price"}]}'
        ),
        model=MODEL_NAME,
        duration_seconds=2.4,
        used_schema=True,
    )
    at = run_app(monkeypatch, READY, generation=generation)
    at.button[0].click().run()
    assert not at.exception
    text = markdown_text(at)
    assert "No usable test inputs" in text
    assert "not a boolean" in text
    assert "Observed behaviour" not in text


def test_curated_baseline_works_offline_and_is_labelled(monkeypatch):
    at = run_app(monkeypatch, OFFLINE)
    assert at.button[1].label == "Run curated baseline"
    at.button[1].click().run()
    assert not at.exception
    text = markdown_text(at)
    assert "Observed behaviour" in text
    assert "Curated baseline" in text
    assert "pre-written cases · not AI-generated" in text
    assert "Why this baseline case exists" in text
    assert "5 cases executed" in text
    assert "Curated baseline · not AI-generated" in text
    assert "Qwen2.5-Coder 7B via local Ollama" not in text
