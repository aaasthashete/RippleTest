"""Deterministic execution and comparison of two trusted function versions.

The engine calls the bundled callables with identical, validated arguments and
records what actually happened. It does not ask the model what it expects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

from .scenarios import Scenario
from .validators import ValidatedCase

if TYPE_CHECKING:
    from .expression import FunctionPair

Classification = Literal["same", "changed", "error"]
Provenance = Literal["ai", "curated"]

# Anything that exposes .original and .modified callables: a trusted Scenario
# or an editable FunctionPair built by the restricted AST interpreter.
RunnableSource = "Scenario | FunctionPair"

# Numeric agreement tolerance. Documented in the README and shown in the UI.
REL_TOL = 1e-9
ABS_TOL = 1e-9


@dataclass(frozen=True)
class Outcome:
    """One real observation: either a returned value or a raised exception."""

    kind: Literal["value", "exception"]
    value: Any = None
    display: str = ""
    exception_type: str | None = None
    exception_message: str | None = None

    @property
    def raised(self) -> bool:
        return self.kind == "exception"


@dataclass(frozen=True)
class CaseResult:
    """The complete, reproducible evidence for one input."""

    number: int
    arguments: dict[str, Any]
    original: Outcome
    modified: Outcome
    classification: Classification
    note: str
    reason: str
    provenance: Provenance
    delta: float | None = None


@dataclass(frozen=True)
class Summary:
    executed: int
    same: int
    changed: int
    errors: int


def observe(func: Callable[..., object], arguments: dict[str, Any]) -> Outcome:
    """Call a trusted function once and record the real result."""
    try:
        value = func(**arguments)
    except Exception as exc:  # A valid input may still raise; that is evidence.
        return Outcome(
            kind="exception",
            display=type(exc).__name__,
            exception_type=type(exc).__name__,
            exception_message=str(exc) or "(no message)",
        )
    return Outcome(kind="value", value=value, display=format_value(value))


def format_value(value: object) -> str:
    """Format a returned value faithfully for display (no invented rounding)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    return repr(value)


def compare(
    original: Outcome, modified: Outcome
) -> tuple[Classification, str, float | None]:
    """Classify the pair of observations.

    Any raised exception is labelled ``error`` for human review, never counted
    as a silent pass. Numeric values agree within a small documented tolerance.
    """
    if original.raised or modified.raised:
        if original.raised and modified.raised:
            if (
                original.exception_type == modified.exception_type
                and original.exception_message == modified.exception_message
            ):
                note = f"Both versions raised {original.exception_type} with the same message."
            else:
                note = "Both versions raised an exception, but not the same one."
        elif original.raised:
            note = f"Only the original version raised {original.exception_type}."
        else:
            note = f"Only the modified version raised {modified.exception_type}."
        return "error", note, None

    both_numbers = _is_number(original.value) and _is_number(modified.value)
    if both_numbers:
        left = float(original.value)
        right = float(modified.value)
        delta = right - left
        if math.isclose(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            return "same", "", 0.0
        return "changed", "", delta

    if type(original.value) is type(modified.value) and original.value == modified.value:
        return "same", "", None

    return "changed", "", None


def run_cases(
    scenario: "Scenario | FunctionPair",
    cases: tuple[ValidatedCase, ...],
    provenance: Provenance,
) -> list[CaseResult]:
    """Execute every validated case against both versions, in order."""
    results: list[CaseResult] = []
    for case in cases:
        arguments = dict(case.arguments)
        original = observe(scenario.original, arguments)
        modified = observe(scenario.modified, arguments)
        classification, note, delta = compare(original, modified)
        results.append(
            CaseResult(
                number=case.number,
                arguments=arguments,
                original=original,
                modified=modified,
                classification=classification,
                note=note,
                reason=case.reason,
                provenance=provenance,
                delta=delta,
            )
        )
    return results


def summarise(results: list[CaseResult] | tuple[CaseResult, ...]) -> Summary:
    """Counts are always derived from executed records, never from model text."""
    return Summary(
        executed=len(results),
        same=sum(1 for result in results if result.classification == "same"),
        changed=sum(1 for result in results if result.classification == "changed"),
        errors=sum(1 for result in results if result.classification == "error"),
    )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
