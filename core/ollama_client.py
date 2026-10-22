"""Real HTTP client for a local Ollama instance.

There is no hosted fallback: every generation request goes to the Ollama URL
below, with the fixed open-weight model. Transport failures are returned as
typed errors so the UI can be honest about them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests

from .scenarios import Scenario

DEFAULT_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5-coder:7b"
HEALTH_TIMEOUT = 3.0
GENERATION_TIMEOUT = 120.0
MIN_CASES = 5
MAX_CASES = 8


@dataclass(frozen=True)
class ModelStatus:
    """Result of a real GET /api/tags health check."""

    state: str  # "available" | "model_missing" | "unavailable"
    model: str
    installed_models: tuple[str, ...] = ()
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.state == "available"


@dataclass(frozen=True)
class GenerationResult:
    """Transport-level result. JSON parsing/validation happens separately."""

    ok: bool
    raw_content: str = ""
    error_kind: str = ""
    error_message: str = ""
    detail: str = ""
    model: str = MODEL_NAME
    duration_seconds: float = 0.0
    used_schema: bool = False
    total_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    notice: str = ""


def check_model(
    base_url: str = DEFAULT_BASE_URL,
    model: str = MODEL_NAME,
    timeout: float = HEALTH_TIMEOUT,
) -> ModelStatus:
    """Ask Ollama which models it has; never assume the answer."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout:
        return ModelStatus(
            "unavailable", model, detail=f"Ollama did not respond within {timeout:g}s at {base_url}."
        )
    except requests.ConnectionError:
        return ModelStatus(
            "unavailable", model, detail=f"Could not reach Ollama at {base_url}."
        )
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", "?")
        return ModelStatus(
            "unavailable", model, detail=f"Ollama health check returned HTTP {status}."
        )
    except requests.RequestException as exc:
        return ModelStatus(
            "unavailable", model, detail=f"Ollama health check failed ({type(exc).__name__})."
        )
    except ValueError:
        return ModelStatus(
            "unavailable", model, detail="Ollama returned an unreadable model list."
        )

    names: list[str] = []
    if isinstance(payload, dict) and isinstance(payload.get("models"), list):
        for entry in payload["models"]:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("model")
                if isinstance(name, str):
                    names.append(name)

    installed = tuple(sorted(set(names)))
    if model in installed:
        return ModelStatus("available", model, installed, detail=f"{model} is ready.")
    return ModelStatus(
        "model_missing",
        model,
        installed,
        detail=f"{model} is not installed. Run `ollama pull {model}` and try again.",
    )


def build_schema(scenario: Scenario) -> dict[str, Any]:
    """JSON Schema for Ollama's structured-output mode, scoped to this scenario."""
    argument_properties = {
        parameter.name: {
            "type": "number",
            "description": (
                f"{parameter.description} Between {parameter.low:g} and {parameter.high:g}, inclusive."
            ),
        }
        for parameter in scenario.parameters
    }

    return {
        "type": "object",
        "properties": {
            "test_cases": {
                "type": "array",
                "minItems": MIN_CASES,
                "maxItems": MAX_CASES,
                "items": {
                    "type": "object",
                    "properties": {
                        "arguments": {
                            "type": "object",
                            "properties": argument_properties,
                            "required": list(scenario.parameter_names),
                            "additionalProperties": False,
                        },
                        "reason": {
                            "type": "string",
                            "description": "One short sentence on why this input is worth checking.",
                        },
                    },
                    "required": ["arguments", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["test_cases"],
        "additionalProperties": False,
    }


def build_messages(scenario: Scenario) -> list[dict[str, str]]:
    """Prompt messages. The model proposes data only; Python does the rest."""
    argument_lines = "\n".join(
        f"- `{parameter.name}`: a number from {parameter.low:g} to {parameter.high:g}. "
        f"{parameter.description}"
        for parameter in scenario.parameters
    )

    system = (
        "You generate test INPUT DATA for a differential testing tool. "
        "You never write code to run, never predict results, and never claim a "
        "bug. Reply with one JSON object only."
    )

    user = f"""You are helping a developer compare two versions of a Python function.

INTENDED BEHAVIOUR
{scenario.contract}

ORIGINAL — v1
```python
{scenario.original_source}```

MODIFIED — v2
```python
{scenario.modified_source}```

ACCEPTED ARGUMENTS
{argument_lines}

TASK
Propose between {MIN_CASES} and {MAX_CASES} test inputs that could reveal a difference
between the two versions.

- Use only the argument names above, with numeric values inside the stated ranges.
- Vary the cases: include ordinary values, zero or extremes, and — when the code
  compares values or has a threshold — at least one exact boundary or equality case.
- Do not include expected outputs. A separate engine executes both versions.
- Keep each reason to one short sentence (under 25 words).

Reply with JSON exactly shaped like:
{{"test_cases": [{{"arguments": {{{", ".join(f'"{name}": 0' for name in scenario.parameter_names)}}}, "reason": "..."}}]}}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_test_cases(
    scenario: Scenario,
    base_url: str = DEFAULT_BASE_URL,
    model: str = MODEL_NAME,
    timeout: float = GENERATION_TIMEOUT,
) -> GenerationResult:
    """POST a real, non-streaming /api/chat request to the local model."""
    url = f"{base_url.rstrip('/')}/api/chat"
    messages = build_messages(scenario)
    schema = build_schema(scenario)
    started = time.perf_counter()
    last_detail = ""

    for attempt in (1, 2):
        use_schema = attempt == 1
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": schema if use_schema else "json",
            "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 900},
            "keep_alive": "5m",
        }

        try:
            response = requests.post(url, json=body, timeout=(5, timeout))
        except requests.Timeout:
            return _failure(
                "timeout",
                f"The local model did not finish within {timeout:g} seconds.",
                f"No response from {url}. The model may still be loading.",
                model,
                time.perf_counter() - started,
                use_schema,
            )
        except requests.ConnectionError:
            return _failure(
                "offline",
                f"Ollama is not reachable at {base_url}.",
                "Start Ollama (`ollama serve`) and confirm the model is installed.",
                model,
                time.perf_counter() - started,
                use_schema,
            )
        except requests.RequestException as exc:
            return _failure(
                "transport",
                "The request to Ollama failed.",
                f"{type(exc).__name__}: {exc}",
                model,
                time.perf_counter() - started,
                use_schema,
            )

        elapsed = time.perf_counter() - started
        response_text = response.text or ""
        lowered = response_text.lower()

        if response.status_code == 404 or ("model" in lowered and "not found" in lowered):
            return _failure(
                "model_missing",
                f"Ollama does not have {model} installed.",
                response_text[:400] or f"Run `ollama pull {model}`.",
                model,
                elapsed,
                use_schema,
            )

        if not response.ok:
            last_detail = f"HTTP {response.status_code}: {response_text[:300]}"
            # A schema the runtime rejects should not hide a real request behind
            # it: retry once in plain JSON mode, then validate just as strictly.
            if use_schema and response.status_code in {400, 422, 500}:
                continue
            return _failure(
                "http_error",
                f"Ollama returned HTTP {response.status_code}.",
                last_detail,
                model,
                elapsed,
                use_schema,
            )

        try:
            payload = response.json()
        except ValueError:
            return _failure(
                "bad_response",
                "Ollama returned a response that was not JSON.",
                response_text[:300],
                model,
                elapsed,
                use_schema,
            )

        content = payload.get("message", {}).get("content") if isinstance(payload, dict) else None
        if not isinstance(content, str) or not content.strip():
            return _failure(
                "empty_response",
                "Ollama returned an empty message.",
                json.dumps(payload)[:300],
                model,
                elapsed,
                use_schema,
            )

        return GenerationResult(
            ok=True,
            raw_content=content,
            model=model,
            duration_seconds=elapsed,
            used_schema=use_schema,
            total_duration_ns=payload.get("total_duration"),
            prompt_eval_count=payload.get("prompt_eval_count"),
            eval_count=payload.get("eval_count"),
            notice=(
                ""
                if use_schema
                else "Ollama rejected the requested JSON schema, so the request was "
                "retried in plain JSON mode. Every case is still validated here."
            ),
        )

    return _failure(
        "http_error",
        "Ollama could not process the request.",
        last_detail,
        model,
        time.perf_counter() - started,
        False,
    )


def _failure(
    kind: str,
    message: str,
    detail: str,
    model: str,
    duration: float,
    used_schema: bool,
) -> GenerationResult:
    return GenerationResult(
        ok=False,
        error_kind=kind,
        error_message=message,
        detail=detail,
        model=model,
        duration_seconds=duration,
        used_schema=used_schema,
    )
