"""Trusted demonstration scenarios.

Both versions of every function below are ordinary, pre-reviewed module-level
definitions. RippleTest imports these callables directly. It never compiles or
executes source text supplied at runtime by a user or by the model.

The `*_source` strings used by the UI are derived from these same functions
(via inspect) so the displayed code cannot drift away from what runs. A test
in `tests/test_scenarios.py` enforces that correspondence.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping


# ---------------------------------------------------------------------------
# Trusted implementations — the only code RippleTest ever executes.
# ---------------------------------------------------------------------------


def _calculate_discount_original(price, discount):
    return price - (price * discount / 100)


def _calculate_discount_modified(price, discount):
    return price - discount


def _shipping_fee_original(cart_total, threshold):
    return 0 if cart_total >= threshold else 50


def _shipping_fee_modified(cart_total, threshold):
    return 0 if cart_total > threshold else 50


# ---------------------------------------------------------------------------
# Scenario metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameter:
    """One accepted argument: a finite number inside an inclusive range."""

    name: str
    low: float
    high: float
    description: str


@dataclass(frozen=True)
class CuratedCase:
    """A pre-written baseline input, used only when labelled as such."""

    arguments: dict
    note: str


@dataclass(frozen=True)
class Scenario:
    id: str
    label: str
    function_name: str
    contract: str
    parameters: tuple[Parameter, ...]
    original_source: str
    modified_source: str
    original: Callable[..., object]
    modified: Callable[..., object]
    curated_cases: tuple[CuratedCase, ...]

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def constraints_text(self) -> str:
        """Human-readable bounds, e.g. ``price 0–10000 · discount 0–100``."""
        parts = [
            f"{parameter.name} {_format_bound(parameter.low)}–{_format_bound(parameter.high)}"
            for parameter in self.parameters
        ]
        return " · ".join(parts)


def _format_bound(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _display_source(func: Callable[..., object], public_name: str) -> str:
    """Return real function source with its private name replaced by the
    public demo name, so both panes read like the reviewed snippet."""
    source = inspect.getsource(func)

    # Read-only AST check: the text we are about to display must be exactly one
    # function definition. This never compiles or executes the source.
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise AssertionError(f"Unexpected source shape for {func.__name__!r}.")
    if tree.body[0].name != func.__name__:
        raise AssertionError(f"Source/callable mismatch for {func.__name__!r}.")

    lines = source.splitlines()
    private_def = f"def {func.__name__}("
    if not lines or not lines[0].lstrip().startswith(private_def):
        raise AssertionError(
            f"Unexpected source layout for {func.__name__!r}; expected a module-level def."
        )
    lines[0] = lines[0].replace(private_def, f"def {public_name}(", 1)
    return "\n".join(lines).strip() + "\n"


_DISCOUNT = Scenario(
    id="discount",
    label="Percentage discount",
    function_name="calculate_discount",
    contract=(
        "`discount` is a percentage taken off the price, so 10% off 1000 should "
        "return 900. The modified version subtracts the discount as a flat "
        "currency amount instead."
    ),
    parameters=(
        Parameter("price", 0, 10_000, "The price before any discount."),
        Parameter("discount", 0, 100, "A percentage, not a currency amount."),
    ),
    original_source=_display_source(_calculate_discount_original, "calculate_discount"),
    modified_source=_display_source(_calculate_discount_modified, "calculate_discount"),
    original=_calculate_discount_original,
    modified=_calculate_discount_modified,
    curated_cases=(
        CuratedCase(
            {"price": 100, "discount": 10},
            "A 10% discount on a small price — both versions agree here.",
        ),
        CuratedCase(
            {"price": 1000, "discount": 10},
            "A 10% discount where flat subtraction is visibly different.",
        ),
        CuratedCase(
            {"price": 0, "discount": 5},
            "Zero price: the modified version can go negative.",
        ),
        CuratedCase(
            {"price": 100, "discount": 100},
            "A full discount, where both formulas meet at zero.",
        ),
        CuratedCase(
            {"price": 250, "discount": 25},
            "A fractional result that exposes the formula difference.",
        ),
    ),
)


_SHIPPING = Scenario(
    id="shipping",
    label="Free-shipping boundary",
    function_name="shipping_fee",
    contract=(
        "Shipping is free when the cart total is at least the threshold; "
        "otherwise the fee is a fixed 50. At cart_total 500 and threshold 500, "
        "shipping should be free."
    ),
    parameters=(
        Parameter("cart_total", 0, 10_000, "The value of the cart."),
        Parameter("threshold", 0, 10_000, "The free-shipping threshold."),
    ),
    original_source=_display_source(_shipping_fee_original, "shipping_fee"),
    modified_source=_display_source(_shipping_fee_modified, "shipping_fee"),
    original=_shipping_fee_original,
    modified=_shipping_fee_modified,
    curated_cases=(
        CuratedCase(
            {"cart_total": 400, "threshold": 500},
            "Below the threshold, both versions charge the fee.",
        ),
        CuratedCase(
            {"cart_total": 500, "threshold": 500},
            "Exactly at the threshold — the boundary this edit moves.",
        ),
        CuratedCase(
            {"cart_total": 600, "threshold": 500},
            "Above the threshold, both versions ship free.",
        ),
        CuratedCase(
            {"cart_total": 0, "threshold": 0},
            "Zero total against a zero threshold.",
        ),
    ),
)


SCENARIOS: Mapping[str, Scenario] = MappingProxyType(
    {scenario.id: scenario for scenario in (_DISCOUNT, _SHIPPING)}
)
DEFAULT_SCENARIO_ID = "discount"


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        raise KeyError(f"Unknown scenario: {scenario_id!r}") from exc
