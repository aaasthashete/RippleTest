"""Parse and validate AI-proposed test data.

The model's reply is untrusted input. Nothing here executes code: it only
turns a message from Ollama into validated numeric arguments, rejecting
anything that is not exactly what the selected scenario accepts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from .scenarios import Scenario

MAX_CASES = 8
MAX_REASON_CHARS = 400


class RawTestCase(BaseModel):
    """Structural shape of one proposal. Numeric rules are checked separately."""

    model_config = ConfigDict(extra="forbid")

    arguments: dict[str, Any]
    reason: str = ""


class ModelResponseError(Exception):
    """The reply could not be read as the expected {test_cases: [...]} envelope."""

    def __init__(self, kind: str, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.detail = detail


@dataclass(frozen=True)
class ValidatedCase:
    """A proposal that passed every check and may be executed."""

    number: int
    arguments: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class RejectedCase:
    """A proposal that was refused, with the reason a human can audit."""

    number: int
    raw: str
    reason: str


@dataclass(frozen=True)
class ValidationReport:
    accepted: tuple[ValidatedCase, ...]
    rejected: tuple[RejectedCase, ...]

    @property
    def proposed_count(self) -> int:
        return len(self.accepted) + len(self.rejected)


def parse_json_content(content: str) -> Any:
    """Decode the model message as JSON.

    Structured output should already be clean JSON; stripping an enclosing
    Markdown fence is a tolerance, not a substitute for validation.
    """
    if not isinstance(content, str) or not content.strip():
        raise ModelResponseError("empty_response", "The model returned an empty message.")

    text = content.strip().lstrip("\ufeff").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # drop the opening fence (e.g. ```json)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelResponseError(
            "invalid_json",
            "The model reply was not valid JSON.",
            detail=f"{exc.msg} at line {exc.lineno}, column {exc.colno}.",
        ) from exc


def validate_cases(scenario: Scenario, payload: object) -> ValidationReport:
    """Validate a decoded model reply against one scenario's exact contract."""
    if not isinstance(payload, dict):
        raise ModelResponseError(
            "invalid_structure", "The reply was valid JSON but not a JSON object."
        )

    raw_cases = payload.get("test_cases")
    if not isinstance(raw_cases, list):
        raise ModelResponseError(
            "invalid_structure", 'The reply did not contain a "test_cases" list.'
        )
    if not raw_cases:
        raise ModelResponseError("no_cases", "The model returned an empty list of test cases.")

    accepted: list[ValidatedCase] = []
    rejected: list[RejectedCase] = []
    seen: dict[tuple[float, ...], int] = {}

    for position, raw_case in enumerate(raw_cases, start=1):
        if position > MAX_CASES:
            rejected.append(
                RejectedCase(
                    position,
                    _shorten(raw_case),
                    f"Beyond the {MAX_CASES}-case limit; only the first {MAX_CASES} "
                    "proposals are considered.",
                )
            )
            continue

        try:
            case = RawTestCase.model_validate(raw_case)
        except ValidationError as exc:
            rejected.append(
                RejectedCase(position, _shorten(raw_case), f"Malformed proposal: {_first_error(exc)}")
            )
            continue

        problem = _argument_problem(scenario, case.arguments)
        if problem is not None:
            rejected.append(RejectedCase(position, _shorten(raw_case), problem))
            continue

        arguments = {name: case.arguments[name] for name in scenario.parameter_names}
        duplicate_key = tuple(float(arguments[name]) for name in scenario.parameter_names)
        if duplicate_key in seen:
            rejected.append(
                RejectedCase(
                    position,
                    _shorten(raw_case),
                    f"Duplicate of case #{seen[duplicate_key]}.",
                )
            )
            continue

        seen[duplicate_key] = position
        accepted.append(
            ValidatedCase(
                number=position,
                arguments=arguments,
                reason=_clean_reason(case.reason),
            )
        )

    return ValidationReport(accepted=tuple(accepted), rejected=tuple(rejected))


def _argument_problem(scenario: Scenario, arguments: dict[str, Any]) -> str | None:
    """Return a human-readable reason to reject, or None when the args are safe."""
    expected = scenario.parameter_names
    missing = [name for name in expected if name not in arguments]
    extra = [name for name in arguments if name not in expected]

    if missing:
        return f"Missing argument(s): {', '.join(missing)}."
    if extra:
        return f"Unsupported argument(s): {', '.join(extra)}."

    for parameter in scenario.parameters:
        value = arguments[parameter.name]

        # bool is a subclass of int in Python; it is never a valid test input.
        if isinstance(value, bool):
            return f"{parameter.name} must be a number, not a boolean."
        if not isinstance(value, (int, float)):
            return f"{parameter.name} must be a number, not {type(value).__name__}."

        try:
            number = float(value)
        except (OverflowError, ValueError):
            return f"{parameter.name} is too large to evaluate safely."

        if not math.isfinite(number):
            return f"{parameter.name} must be a finite number."
        if not (parameter.low <= number <= parameter.high):
            return (
                f"{parameter.name}={number:g} is outside the allowed range "
                f"[{parameter.low:g}, {parameter.high:g}]."
            )

    return None


def _first_error(exc: ValidationError) -> str:
    """A compact one-line summary of the first Pydantic problem."""
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error.get("loc", ()))
    message = error.get("msg", "invalid value")
    prefix = f"{location}: " if location else ""
    text = f"{prefix}{message}"
    return text if len(text) <= 160 else text[:157] + "…"


def _shorten(raw_case: object, limit: int = 180) -> str:
    try:
        text = json.dumps(raw_case, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(raw_case)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clean_reason(reason: str) -> str:
    text = " ".join(str(reason).split())
    if not text:
        return "No rationale provided."
    return text if len(text) <= MAX_REASON_CHARS else text[: MAX_REASON_CHARS - 1] + "…"
