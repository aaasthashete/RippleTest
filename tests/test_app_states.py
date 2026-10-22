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


def test_page_renders_editable_panels_and_the_real_status(monkeypatch):
    at = run_app(monkeypatch, READY)
    assert not at.exception
    text = markdown_text(at)
    assert "Small change." in text
    assert "Massive consequences." in text
    assert "Local / Qwen2.5-Coder 7B · Ready" in text
    assert "Original implementation" in text
    assert "Modified implementation" in text
    assert "Analyse change" in [button.label for button in at.button]
    assert "No analysis yet." in text
    assert "Ollama unavailable" not in text
    # The editors are seeded with the bundled discount example.
    areas = {area.label: area.value for area in at.text_area}
    assert "calculate_discount" in areas["Original implementation source"]
    assert "price - discount" in areas["Modified implementation source"]


def test_editors_hold_independent_editable_text(monkeypatch):
    """The two panels are real text areas a developer can type into."""
    at = run_app(monkeypatch, READY)
    original = at.text_area[0]
    modified = at.text_area[1]
    original.set_value("def f(x):\n    return x * 2\n")
    modified.set_value("def f(x):\n    return x * 3\n")
    at.run()
    assert not at.exception
    assert at.text_area[0].value == "def f(x):\n    return x * 2\n"
    assert at.text_area[1].value == "def f(x):\n    return x * 3\n"


def test_offline_state_is_actionable_and_shows_no_results(monkeypatch):
    at = run_app(monkeypatch, OFFLINE)
    assert not at.exception
    text = markdown_text(at)
    assert "Ollama / offline" in text
    assert "Ollama unavailable" in text
    assert "No hosted model will be substituted." in text
    assert "No analysis yet." in text
    assert "Observed behaviour" not in text


def test_model_missing_state_names_the_pull_command(monkeypatch):
    status = ModelStatus("model_missing", MODEL_NAME, ("llama3:8b",), detail="not installed")
    at = run_app(monkeypatch, status)
    assert not at.exception
    text = markdown_text(at)
    assert "Ollama / model missing" in text
    assert "ollama pull qwen2.5-coder:7b" in text


def test_unsupported_python_is_refused_visibly_and_never_executed(monkeypatch):
    """Pasting arbitrary Python must produce an explanation, not execution."""
    at = run_app(monkeypatch, READY)
    at.text_area[1].set_value("import os\n\ndef calculate_discount(price, discount):\n    return os.getcwd()\n")
    at.run()
    analyse = next(b for b in at.button if b.label == "Analyse change")
    analyse.click().run()
    assert not at.exception
    text = markdown_text(at)
    assert "Unsupported Python in the editor" in text
    assert "not executed" in text
    assert "Observed behaviour" not in text
    assert "No analysis yet." not in text


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
    analyse = next(b for b in at.button if b.label == "Analyse change")
    analyse.click().run()
    assert not at.exception
    text = markdown_text(at)
    assert "Ollama is offline" in text
    assert "Observed behaviour" not in text
    assert "No analysis yet." not in text


def test_invalid_json_reply_is_reported_as_ai_response_invalid(monkeypatch):
    generation = GenerationResult(
        ok=True,
        raw_content="Sure! Here is a test plan, not JSON.",
        model=MODEL_NAME,
        duration_seconds=2.2,
        used_schema=True,
    )
    at = run_app(monkeypatch, READY, generation=generation)
    next(b for b in at.button if b.label == "Analyse change").click().run()
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
    next(b for b in at.button if b.label == "Analyse change").click().run()
    assert not at.exception
    text = markdown_text(at)
    assert "No usable test inputs" in text
    assert "not a boolean" in text
    assert "Observed behaviour" not in text


def test_edited_code_renders_results_from_the_interpreter(monkeypatch):
    """Regression: an editable-code run must render, not be discarded as stale.

    The edited text (price - price*discount) produces a large negative swing that
    only appears if the interpreter actually ran the developer's code.
    """
    generation = GenerationResult(
        ok=True,
        raw_content=(
            '{"test_cases": ['
            '{"arguments": {"price": 1000, "discount": 10}, "reason": "typical"},'
            '{"arguments": {"price": 500, "discount": 2}, "reason": "small"}'
            ']}'
        ),
        model=MODEL_NAME,
        duration_seconds=3.0,
        used_schema=True,
    )
    at = run_app(monkeypatch, READY, generation=generation)
    at.text_area[1].set_value(
        "def calculate_discount(price, discount):\n    return price - (price * discount)\n"
    )
    at.run()
    next(b for b in at.button if b.label == "Analyse change").click().run()
    assert not at.exception
    text = markdown_text(at)
    assert "Observed behaviour" in text
    assert "Restricted AST interpreter" in text
    assert "Trusted bundled function" not in text
    # 1000 - (1000 * 10) = -9000 proves the edited expression, not the bundled
    # example, was executed.
    assert "-9000" in text


def test_unedited_example_uses_the_trusted_engine(monkeypatch):
    """The reviewed bundled path stays available and is labelled honestly."""
    generation = GenerationResult(
        ok=True,
        raw_content=(
            '{"test_cases": [{"arguments": {"price": 1000, "discount": 10}, '
            '"reason": "typical"}]}'
        ),
        model=MODEL_NAME,
        duration_seconds=2.0,
        used_schema=True,
    )
    at = run_app(monkeypatch, READY, generation=generation)
    next(b for b in at.button if b.label == "Analyse change").click().run()
    assert not at.exception
    text = markdown_text(at)
    assert "Trusted bundled function" in text
    assert "Restricted AST interpreter" not in text


def test_curated_baseline_works_offline_and_is_labelled(monkeypatch):
    at = run_app(monkeypatch, OFFLINE)
    baseline = next(b for b in at.button if b.label == "Run curated baseline")
    assert baseline.disabled is False  # editors hold the matching bundled example
    baseline.click().run()
    assert not at.exception
    text = markdown_text(at)
    assert "Observed behaviour" in text
    assert "Curated baseline" in text
    assert "pre-written" in text
    assert "Why this baseline case exists" in text
    assert "5 cases executed" in text
    assert "Curated baseline · not AI-generated" in text
    assert "Qwen2.5-Coder 7B via local Ollama" not in text


def test_baseline_is_disabled_for_unmatched_custom_code(monkeypatch):
    """Curated cases belong to a bundled example; never run them on other code."""
    at = run_app(monkeypatch, OFFLINE)
    at.text_area[1].set_value("def calculate_discount(price, discount):\n    return price\n")
    at.run()
    baseline = next(b for b in at.button if b.label == "Run curated baseline")
    assert baseline.disabled is True


def test_editing_after_successful_ai_run_clears_outdated_evidence(monkeypatch):
    """Never show outputs calculated for a previous version of editor content."""
    generation = GenerationResult(
        ok=True,
        raw_content=(
            '{"test_cases": [{"arguments": {"price": 1000, "discount": 10}, '
            '"reason": "normal"}]}'
        ),
        model=MODEL_NAME,
        duration_seconds=1.0,
        used_schema=True,
    )
    at = run_app(monkeypatch, READY, generation=generation)
    next(b for b in at.button if b.label == "Analyse change").click().run()
    assert not at.exception
    assert "Observed behaviour" in markdown_text(at)
    at.text_area[1].set_value(
        "def calculate_discount(price, discount):\n"
        "    return price - (price * discount / 200)\n"
    ).run()
    assert not at.exception
    text = markdown_text(at)
    assert "Observed behaviour" not in text
    assert "No analysis yet." in text
