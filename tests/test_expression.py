"""Tests for the restricted AST interpreter that powers editable code.

The safety promise is: no exec, no eval, no imports, no function calls, and no
host code execution. These tests assert both the allowed arithmetic semantics
and that hostile or unsupported input is refused with a readable explanation.
"""

import math

import pytest

from core.expression import (
    CUSTOM_PARAMETER_HIGH,
    CUSTOM_PARAMETER_LOW,
    ENGINE_INTERPRETER,
    ENGINE_TRUSTED,
    ENGINE_LABELS,
    EditError,
    build_callable,
    build_pair,
    default_parameters,
    parse_function,
)
from core.pipeline import build_run
from core.validators import ValidationReport, ValidatedCase

DISCOUNT_ORIGINAL = "def calculate_discount(price, discount):\n    return price - (price * discount / 100)\n"
DISCOUNT_MODIFIED = "def calculate_discount(price, discount):\n    return price - discount\n"
SHIPPING_ORIGINAL = "def shipping_fee(cart_total, threshold):\n    return 0 if cart_total >= threshold else 50\n"
SHIPPING_MODIFIED = "def shipping_fee(cart_total, threshold):\n    return 0 if cart_total > threshold else 50\n"


# --- Parsing and signature checks ------------------------------------------


def test_parses_parameters_and_function_name():
    parsed = parse_function(DISCOUNT_ORIGINAL)
    assert parsed.name == "calculate_discount"
    assert parsed.parameters == ("price", "discount")


def test_pair_exposes_a_scenario_compatible_surface():
    pair = build_pair(DISCOUNT_ORIGINAL, DISCOUNT_MODIFIED)
    assert pair.id == "custom"
    assert pair.label == "Custom code"
    assert pair.parameter_names == ("price", "discount")
    assert pair.function_name == "calculate_discount"
    assert "No stated contract" in pair.contract
    assert pair.parameters == default_parameters(("price", "discount"))


def test_pair_requires_matching_parameter_names_and_order():
    with pytest.raises(EditError) as info:
        build_pair(DISCOUNT_ORIGINAL, "def calculate_discount(price):\n    return price\n")
    assert "same parameters" in info.value.message
    assert "price, discount" in str(info.value)


def test_empty_and_whitespace_panels_are_rejected():
    for source in ("", "   \n  ", None):
        with pytest.raises(EditError):
            parse_function(source)


def test_bundled_examples_execute_identically_to_the_interpreter():
    """The editable path must reproduce the documented truth table."""
    pair = build_pair(DISCOUNT_ORIGINAL, DISCOUNT_MODIFIED)
    assert pair.original(price=100, discount=10) == pytest.approx(90.0)
    assert pair.modified(price=100, discount=10) == pytest.approx(90)
    assert pair.original(price=1000, discount=10) == pytest.approx(900.0)
    assert pair.modified(price=1000, discount=10) == pytest.approx(990)
    assert pair.original(price=100, discount=100) == pytest.approx(0.0)
    assert pair.original(price=250, discount=25) == pytest.approx(187.5)


def test_ternary_and_comparison_semantics_match_python():
    pair = build_pair(SHIPPING_ORIGINAL, SHIPPING_MODIFIED)
    assert pair.original(cart_total=400, threshold=500) == 50
    assert pair.original(cart_total=500, threshold=500) == 0
    assert pair.modified(cart_total=500, threshold=500) == 50
    assert pair.modified(cart_total=600, threshold=500) == 0
    assert pair.original(cart_total=0, threshold=0) == 0
    assert pair.modified(cart_total=0, threshold=0) == 50


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("1 + 2", 3.0),
        ("10 - 3", 7.0),
        ("4 * 5", 20.0),
        ("10 / 4", 2.5),
        ("10 // 4", 2.0),
        ("10 % 4", 2.0),
        ("2 ** 3", 8.0),
        ("-3 + 1", -2.0),
        ("+3", 3.0),
        ("1 if 2 > 1 else 0", 1.0),
        ("(1 + 2) * 3", 9.0),
    ],
)
def test_operator_semantics(expression, expected):
    source = f"def f(x):\n    return {expression}\n"
    callable_ = build_callable(parse_function(source))
    assert callable_(x=1) == pytest.approx(expected)


def test_and_or_conditions_work_even_though_boolean_returns_are_refused():
    source = "def f(x):\n    return x if (x > 5 and x < 100) else -1\n"
    callable_ = build_callable(parse_function(source))
    assert callable_(x=10) == 10.0
    assert callable_(x=200) == -1.0


def test_boolean_return_is_refused():
    source = "def f(x):\n    return x > 5\n"
    with pytest.raises(TypeError, match="boolean"):
        build_callable(parse_function(source))(x=10)


def test_real_exceptions_are_raised_for_capture():
    pair = build_pair("def f(x):\n    return 10 / x\n", "def f(x):\n    return 10 // x\n")
    with pytest.raises(ZeroDivisionError):
        pair.original(x=0)
    with pytest.raises(ZeroDivisionError):
        pair.modified(x=0)


def test_non_finite_result_is_refused():
    source = "def f(x):\n    return x * 1e308\n"
    with pytest.raises(ValueError, match="non-finite"):
        build_callable(parse_function(source))(x=1e308)


def test_missing_argument_is_a_type_error():
    pair = build_pair(DISCOUNT_ORIGINAL, DISCOUNT_MODIFIED)
    with pytest.raises(TypeError, match="Missing argument"):
        pair.original(price=100)


# --- Rejections: nothing unsafe reaches the interpreter ---------------------


@pytest.mark.parametrize(
    "source,fragment",
    [
        ("def f(x):\n    return open('/etc/passwd').read()\n", "Function calls are not supported"),
        ("def f(x):\n    return eval('1')\n", "Function calls are not supported"),
        ("def f(x):\n    return x.__class__\n", "Attribute access"),
        ("def f(x):\n    return x[0]\n", "Indexing"),
        ("def f(x):\n    return [1, 2]\n", "List literals"),
        ("def f(x):\n    return {'a': 1}\n", "Dict literals"),
        ("def f(x):\n    return f'{x}'\n", "f-strings"),
        ("def f(x):\n    return lambda: x\n", "Lambda"),
        ("def f(x):\n    return [i for i in range(3)]\n", "List comprehensions"),
        ("def f(x):\n    return (y := x)\n", "Walrus"),
        ("import os\ndef f(x):\n    return x\n", "exactly one function definition"),
        ("def f(x):\n    y = x * 2\n    return y\n", "single return statement"),
        ("def f(x):\n    pass\n", "single return statement"),
        ("def f(x):\n    for i in (1, 2):\n        pass\n    return x\n", "single return statement"),
        ("x = 5\n", "single top-level function definition"),
        ("def f(x):\n    return 'text'\n", "finite numeric constants"),
        ("def f(x):\n    return None\n", "finite numeric constants"),
        ("@decorator\ndef f(x):\n    return x\n", "Decorators"),
        ("async def f(x):\n    return x\n", "Async functions"),
        ("def f(x=1):\n    return x\n", "Default parameter values"),
        ("def f(*args):\n    return 1\n", "*args and **kwargs"),
        ("def f(x):\n    return unknown_name\n", "is not a parameter"),
        ("def f(x):\n    return x\n\n\ndef g(y):\n    return y\n", "exactly one function definition"),
        ("while True:\n    pass\n", "single top-level function definition"),
        ("def f(x):\n    return x\n", None),  # valid control case
    ],
)
def test_unsupported_syntax_is_refused_with_a_reason(source, fragment):
    if fragment is None:
        build_pair(source, source)  # control: must not raise
        return
    with pytest.raises(EditError) as info:
        build_pair(source, source)
    assert fragment.lower() in str(info.value).lower()


def test_edit_error_carries_line_number_when_known():
    with pytest.raises(EditError) as info:
        parse_function("def f(x):\n    return open('x')\n")
    assert info.value.line == 2
    assert "Line 2" in str(info.value)


def test_syntax_errors_are_reported_readably():
    with pytest.raises(EditError) as info:
        parse_function("def f(x)\n    return x\n")
    assert "not valid Python" in info.value.message


def test_exponent_magnitude_is_capped():
    pair = build_pair("def f(x):\n    return x ** 2\n", "def f(x):\n    return x ** 3\n")
    assert pair.original(x=4) == 16.0
    with pytest.raises(ValueError, match="Exponent"):
        build_pair("def f(x):\n    return x ** 10000\n", "def f(x):\n    return x ** 2\n").original(x=2)


def test_deeply_nested_expressions_are_refused():
    nested = "1 " + "+ 1 " * 60
    with pytest.raises(EditError, match="nodes|nested"):
        parse_function(f"def f(x):\n    return {nested}\n")


def test_oversized_source_is_refused():
    with pytest.raises(EditError, match="characters"):
        parse_function("def f(x):\n    return " + " + ".join(["1"] * 3000) + "\n")


def test_too_many_parameters_are_refused():
    names = ", ".join(f"p{i}" for i in range(10))
    with pytest.raises(EditError, match="parameters"):
        parse_function(f"def f({names}):\n    return p0\n")


def test_default_parameter_bounds_are_symmetric_and_finite():
    params = default_parameters(("a", "b"))
    assert all(p.low == CUSTOM_PARAMETER_LOW and p.high == CUSTOM_PARAMETER_HIGH for p in params)
    assert all(math.isfinite(p.low) and math.isfinite(p.high) for p in params)


def test_engine_labels_are_distinct_and_descriptive():
    assert ENGINE_TRUSTED != ENGINE_INTERPRETER
    assert "Trusted bundled" in ENGINE_LABELS[ENGINE_TRUSTED]
    assert "interpreter" in ENGINE_LABELS[ENGINE_INTERPRETER].lower()


# --- The editable path feeds the real differential engine -------------------


def test_editable_pair_runs_through_the_real_engine():
    """End to end: edited code -> validated inputs -> real comparison."""
    pair = build_pair(DISCOUNT_ORIGINAL, DISCOUNT_MODIFIED)
    cases = tuple(
        ValidatedCase(number=i, arguments=args, reason=f"case {i}")
        for i, args in enumerate(
            [
                {"price": 1000, "discount": 10},
                {"price": 100, "discount": 10},
                {"price": 250, "discount": 25},
            ],
            start=1,
        )
    )
    run = build_run(pair, ValidationReport(accepted=cases, rejected=()), provenance="ai")
    assert run.summary.executed == 3
    assert run.summary.changed == 2
    assert run.summary.same == 1
    assert run.results[0].original.display == "900.0"
    assert run.results[0].modified.display == "990"


def test_editing_the_modified_code_changes_what_the_engine_reports():
    """The core editability promise: edited text must change the outcome."""
    edited = "def calculate_discount(price, discount):\n    return price - (price * discount)\n"
    pair = build_pair(DISCOUNT_ORIGINAL, edited)
    cases = (ValidatedCase(1, {"price": 1000, "discount": 10}, "case"),)
    run = build_run(pair, ValidationReport(accepted=cases, rejected=()), provenance="ai")
    result = run.results[0]
    assert result.classification == "changed"
    assert result.original.display == "900.0"
    # 1000 - (1000 * 10) = -9000, which is only possible because the edit was interpreted.
    assert result.modified.display == "-9000"


def test_interpreter_matches_trusted_execution_for_same_source():
    """Sanity: our interpreter is not silently diverging from real Python."""
    pair = build_pair(SHIPPING_ORIGINAL, SHIPPING_MODIFIED)
    for cart_total in (0, 499, 500, 501, 10_000):
        for threshold in (0, 500, 10_000):
            args = {"cart_total": cart_total, "threshold": threshold}
            expected_original = 0 if cart_total >= threshold else 50
            expected_modified = 0 if cart_total > threshold else 50
            assert pair.original(**args) == expected_original
            assert pair.modified(**args) == expected_modified
