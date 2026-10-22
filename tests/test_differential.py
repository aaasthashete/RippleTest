import dataclasses
import json

import pytest

from core.differential import (
    Outcome,
    compare,
    observe,
    run_cases,
    summarise,
)
from core.pipeline import build_run, run_curated_baseline
from core.report import run_report
from core.scenarios import get_scenario
from core.validators import ValidatedCase, ValidationReport

DISCOUNT = get_scenario("discount")
SHIPPING = get_scenario("shipping")


def test_curated_baseline_classifies_the_documented_truth_table():
    run = run_curated_baseline(DISCOUNT)
    by_input = {tuple(result.arguments.values()): result for result in run.results}

    assert by_input[(100, 10)].classification == "same"
    assert by_input[(1000, 10)].classification == "changed"
    assert by_input[(0, 5)].classification == "changed"
    assert by_input[(100, 100)].classification == "same"
    assert by_input[(250, 25)].classification == "changed"

    assert by_input[(1000, 10)].original.display == "900.0"
    assert by_input[(1000, 10)].modified.display == "990"
    assert by_input[(1000, 10)].delta == pytest.approx(90.0)


def test_shipping_boundary_is_detected_as_changed():
    run = run_curated_baseline(SHIPPING)
    results = {tuple(result.arguments.values()): result for result in run.results}

    assert results[(500, 500)].classification == "changed"
    assert (results[(500, 500)].original.display, results[(500, 500)].modified.display) == ("0", "50")
    assert results[(400, 500)].classification == "same"
    assert results[(600, 500)].classification == "same"
    assert results[(0, 0)].classification == "changed"


def test_summary_counts_are_derived_from_records():
    run = run_curated_baseline(DISCOUNT)
    summary = run.summary
    assert summary.executed == len(run.results) == 5
    assert summary.same + summary.changed + summary.errors == summary.executed
    assert summary.changed == 3
    assert summary.same == 2
    assert summary.errors == 0


def test_repeated_execution_is_deterministic():
    first = run_curated_baseline(DISCOUNT)
    second = run_curated_baseline(DISCOUNT)
    assert [(r.number, r.original.display, r.modified.display, r.classification) for r in first.results] == [
        (r.number, r.original.display, r.modified.display, r.classification) for r in second.results
    ]


def test_numeric_comparison_uses_documented_tolerance():
    assert compare(Outcome("value", 0.30000000000000004, "0.30000000000000004"),
                   Outcome("value", 0.3, "0.3"))[0] == "same"
    assert compare(Outcome("value", 1.0, "1.0"), Outcome("value", 1.01, "1.01"))[0] == "changed"


def test_non_numeric_values_compare_by_value_and_type():
    assert compare(Outcome("value", "free", "'free'"), Outcome("value", "free", "'free'"))[0] == "same"
    assert compare(Outcome("value", "free", "'free'"), Outcome("value", 0, "0"))[0] == "changed"


def test_exception_is_recorded_as_evidence_without_crashing():
    def boom(**kwargs):
        raise ZeroDivisionError("cannot divide by zero")

    outcome = observe(boom, {"price": 1})
    assert outcome.kind == "exception"
    assert outcome.exception_type == "ZeroDivisionError"
    assert outcome.exception_message == "cannot divide by zero"

    failed = dataclasses.replace(DISCOUNT, original=boom)
    case = ValidatedCase(1, {"price": 100, "discount": 10}, "test")
    result = run_cases(failed, (case,), "ai")[0]

    assert result.classification == "error"
    assert "Only the original version raised ZeroDivisionError" in result.note
    assert summarise([result]).errors == 1


def test_different_exceptions_are_reported_as_error_with_note():
    def left(**kwargs):
        raise ValueError("left")

    def right(**kwargs):
        raise KeyError("right")

    scenario = dataclasses.replace(DISCOUNT, original=left, modified=right)
    case = ValidatedCase(1, {"price": 100, "discount": 10}, "test")
    result = run_cases(scenario, (case,), "ai")[0]
    assert result.classification == "error"
    assert result.note == "Both versions raised an exception, but not the same one."


def test_arguments_are_identical_for_both_versions():
    seen = []

    def left(**kwargs):
        seen.append(("left", dict(kwargs)))
        return 1

    def right(**kwargs):
        seen.append(("right", dict(kwargs)))
        return 2

    scenario = dataclasses.replace(DISCOUNT, original=left, modified=right)
    run_cases(scenario, (ValidatedCase(1, {"price": 42, "discount": 7}, "test"),), "ai")
    assert seen[0][0] == "left" and seen[1][0] == "right"
    assert seen[0][1] == seen[1][1] == {"price": 42, "discount": 7}


def test_build_run_refuses_to_execute_nothing():
    with pytest.raises(ValueError):
        build_run(DISCOUNT, ValidationReport(accepted=(), rejected=()), provenance="ai")


def test_ai_run_keeps_model_provenance_separate_from_curated():
    case = ValidatedCase(1, {"price": 1000, "discount": 10}, "Suggested boundary.")
    run = build_run(
        DISCOUNT,
        ValidationReport(accepted=(case,), rejected=()),
        provenance="ai",
        model="qwen2.5-coder:7b",
        duration_seconds=2.5,
    )
    assert run.provenance == "ai"
    assert run.model == "qwen2.5-coder:7b"
    assert all(result.provenance == "ai" for result in run.results)
    assert run.results[0].reason == "Suggested boundary."

    baseline = run_curated_baseline(DISCOUNT)
    assert baseline.provenance == "curated"
    assert baseline.model is None
    assert all(result.provenance == "curated" for result in baseline.results)


def test_json_report_matches_executed_records_and_is_serialisable():
    run = run_curated_baseline(SHIPPING)
    report = run_report(run)
    payload = json.dumps(report)  # must not raise

    assert report["summary"]["executed"] == len(report["cases"]) == 4
    assert report["summary"]["changed"] == 2
    assert report["input_provenance"] == "Curated baseline (not AI-generated)"
    assert report["model"] == "none (curated baseline)"
    assert report["cases"][1]["arguments"] == {"cart_total": 500, "threshold": 500}
    assert report["cases"][1]["classification"] == "changed"
    assert report["cases"][1]["original"]["display"] == "0"
    assert report["cases"][1]["modified"]["display"] == "50"
    assert "behavioural difference may be intentional" in report["limitations"][0].lower()
    assert len(payload) > 0


def test_report_records_model_metadata_for_ai_runs():
    case = ValidatedCase(1, {"price": 1000, "discount": 10}, "Suggested.")
    run = build_run(
        DISCOUNT,
        ValidationReport(accepted=(case,), rejected=()),
        provenance="ai",
        model="qwen2.5-coder:7b",
        duration_seconds=3.25,
    )
    report = run_report(run)
    assert report["model"] == "qwen2.5-coder:7b"
    assert report["input_provenance"] == "AI-generated by the local model"
    assert report["model_request_seconds"] == 3.25
