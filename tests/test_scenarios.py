import ast
import inspect
from pathlib import Path

import pytest

from core.scenarios import SCENARIOS, get_scenario


def _parse_function(source: str):
    tree = ast.parse(source)
    assert len(tree.body) == 1, "display source must contain exactly one statement"
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def test_registry_contains_exactly_the_two_demo_scenarios():
    assert set(SCENARIOS) == {"discount", "shipping"}
    assert get_scenario("discount").label == "Percentage discount"
    assert get_scenario("shipping").label == "Free-shipping boundary"


def test_unknown_scenario_is_rejected():
    with pytest.raises(KeyError):
        get_scenario("not-a-scenario")


@pytest.mark.parametrize("scenario", list(SCENARIOS.values()), ids=lambda s: s.id)
def test_displayed_source_matches_the_executed_callable(scenario):
    """Guards against UI source drifting away from the trusted implementation."""
    pairs = (
        (scenario.original_source, scenario.original),
        (scenario.modified_source, scenario.modified),
    )
    for source, func in pairs:
        node = _parse_function(source)
        assert node.name == scenario.function_name
        assert [arg.arg for arg in node.args.args] == list(scenario.parameter_names)

        real_source = inspect.getsource(func)
        private_def = f"def {func.__name__}("
        assert real_source.lstrip().startswith(private_def)

        # Recompute the public view from the callable, then compare.
        lines = real_source.splitlines()
        lines[0] = lines[0].replace(private_def, f"def {scenario.function_name}(", 1)
        assert source == "\n".join(lines).strip() + "\n"
        assert func.__name__ not in source
        assert "return" in source


@pytest.mark.parametrize("scenario", list(SCENARIOS.values()), ids=lambda s: s.id)
def test_every_curated_case_is_inside_the_declared_bounds(scenario):
    for case in scenario.curated_cases:
        assert set(case.arguments) == set(scenario.parameter_names)
        for parameter in scenario.parameters:
            value = case.arguments[parameter.name]
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            assert parameter.low <= value <= parameter.high


@pytest.mark.parametrize(
    "price,discount,original,modified",
    [
        (100, 10, 90.0, 90),
        (1000, 10, 900.0, 990),
        (0, 5, 0.0, -5),
        (100, 100, 0.0, 0),
        (250, 25, 187.5, 225),
    ],
)
def test_discount_truth_table(price, discount, original, modified):
    scenario = get_scenario("discount")
    assert scenario.original(price=price, discount=discount) == pytest.approx(original)
    assert scenario.modified(price=price, discount=discount) == pytest.approx(modified)


@pytest.mark.parametrize(
    "cart_total,threshold,original,modified",
    [
        (400, 500, 50, 50),
        (500, 500, 0, 50),
        (600, 500, 0, 0),
        (0, 0, 0, 50),
    ],
)
def test_shipping_truth_table(cart_total, threshold, original, modified):
    scenario = get_scenario("shipping")
    assert scenario.original(cart_total=cart_total, threshold=threshold) == original
    assert scenario.modified(cart_total=cart_total, threshold=threshold) == modified


def test_core_package_has_no_dynamic_code_execution():
    """No eval/exec/compile/import of runtime text anywhere in the engine."""
    forbidden = {"eval", "exec", "compile", "__import__"}
    for path in Path("core").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden, f"{path} calls {node.func.id}()"
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert "subprocess" not in imported, f"{path} imports subprocess"
