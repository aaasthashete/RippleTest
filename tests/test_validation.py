import json

import pytest

from core.scenarios import get_scenario
from core.validators import (
    ModelResponseError,
    parse_json_content,
    validate_cases,
)

DISCOUNT = get_scenario("discount")
SHIPPING = get_scenario("shipping")


def payload(*cases):
    return {"test_cases": list(cases)}


def case(price, discount, reason="A useful case."):
    return {"arguments": {"price": price, "discount": discount}, "reason": reason}


def test_accepts_valid_cases_and_preserves_numbers_and_rationale():
    report = validate_cases(
        DISCOUNT,
        payload(
            case(1000, 10, "Large price, typical percentage."),
            case(0, 5),
            case(100, 100),
            case(250, 25),
            case(99.5, 0.5),
        ),
    )
    assert len(report.accepted) == 5
    assert report.rejected == ()
    assert report.accepted[0].number == 1
    assert report.accepted[0].arguments == {"price": 1000, "discount": 10}
    assert isinstance(report.accepted[0].arguments["price"], int)
    assert report.accepted[0].reason == "Large price, typical percentage."


def test_strips_markdown_code_fence():
    fenced = "```json\n" + json.dumps(payload(case(100, 10))) + "\n```"
    parsed = parse_json_content(fenced)
    assert parsed["test_cases"][0]["arguments"]["price"] == 100


def test_invalid_json_is_reported_with_kind():
    with pytest.raises(ModelResponseError) as info:
        parse_json_content('{"test_cases": [')
    assert info.value.kind == "invalid_json"
    assert info.value.detail


def test_wrong_envelope_is_reported():
    parsed_list = parse_json_content('["price", "discount"]')
    with pytest.raises(ModelResponseError) as info:
        validate_cases(DISCOUNT, parsed_list)
    assert info.value.kind == "invalid_structure"

    with pytest.raises(ModelResponseError) as info:
        validate_cases(DISCOUNT, {"cases": []})
    assert info.value.kind == "invalid_structure"


def test_empty_case_list_is_reported():
    with pytest.raises(ModelResponseError) as info:
        validate_cases(DISCOUNT, {"test_cases": []})
    assert info.value.kind == "no_cases"


@pytest.mark.parametrize(
    "arguments,fragment",
    [
        ({"price": True, "discount": 10}, "not a boolean"),
        ({"price": False, "discount": 10}, "not a boolean"),
        ({"price": "1000", "discount": 10}, "must be a number"),
        ({"price": None, "discount": 10}, "must be a number"),
        ({"price": [1000], "discount": 10}, "must be a number"),
        ({"price": -1, "discount": 10}, "outside the allowed range"),
        ({"price": 10_001, "discount": 10}, "outside the allowed range"),
        ({"price": 10**400, "discount": 10}, "too large to evaluate"),
        ({"price": 100, "discount": 101}, "outside the allowed range"),
        ({"price": 100}, "Missing argument"),
        ({"price": 100, "discount": 10, "tax": 2}, "Unsupported argument"),
        ({"price": 100, "discount": 10, "reason": "x"}, "Unsupported argument"),
    ],
)
def test_rejects_unsafe_or_incompatible_arguments(arguments, fragment):
    report = validate_cases(DISCOUNT, payload({"arguments": arguments, "reason": "test"}))
    assert report.accepted == ()
    assert len(report.rejected) == 1
    assert fragment in report.rejected[0].reason


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_numbers(bad):
    report = validate_cases(DISCOUNT, payload(case(100, bad)))
    assert report.accepted == ()
    assert "finite" in report.rejected[0].reason


def test_rejects_duplicates_but_keeps_the_first():
    report = validate_cases(
        DISCOUNT,
        payload(case(1000, 10), case(500, 5), case(1000, 10, "repeat")),
    )
    assert len(report.accepted) == 2
    assert report.rejected[0].number == 3
    assert "Duplicate of case #1" in report.rejected[0].reason


def test_rejects_duplicate_written_as_float_and_int():
    report = validate_cases(DISCOUNT, payload(case(1000, 10), case(1000.0, 10.0)))
    assert len(report.accepted) == 1
    assert "Duplicate" in report.rejected[0].reason


def test_malformed_case_is_rejected_without_losing_other_cases():
    report = validate_cases(
        DISCOUNT,
        payload(case(100, 10), {"arguments": "price=100"}, case(250, 25)),
    )
    assert [item.number for item in report.accepted] == [1, 3]
    assert report.rejected[0].number == 2
    assert report.rejected[0].reason.startswith("Malformed proposal")


def test_extra_fields_in_a_case_are_rejected():
    report = validate_cases(
        DISCOUNT,
        payload({"arguments": {"price": 100, "discount": 10}, "reason": "x", "expected": 90}),
    )
    assert report.accepted == ()
    assert "Malformed proposal" in report.rejected[0].reason


def test_more_than_eight_cases_are_capped_and_reported():
    cases = [case(price, 10) for price in range(50, 70)]
    report = validate_cases(DISCOUNT, payload(*cases))
    assert len(report.accepted) == 8
    assert len(report.rejected) == 12
    assert all("Beyond the 8-case limit" in item.reason for item in report.rejected)


def test_blank_and_noisy_rationale_is_normalised():
    report = validate_cases(
        DISCOUNT,
        payload(
            case(100, 10, "   "),
            case(200, 20, "First line\n\tsecond   line"),
            case(300, 30, "x" * 900),
        ),
    )
    assert report.accepted[0].reason == "No rationale provided."
    assert report.accepted[1].reason == "First line second line"
    assert len(report.accepted[2].reason) == 400
    assert report.accepted[2].reason.endswith("…")


def test_shipping_scenario_rejects_discount_arguments():
    report = validate_cases(
        SHIPPING,
        payload({"arguments": {"price": 100, "discount": 10}, "reason": "wrong scenario"}),
    )
    assert report.accepted == ()
    assert "Missing argument(s): cart_total, threshold" in report.rejected[0].reason


def test_rejected_raw_payload_is_preserved_for_audit():
    report = validate_cases(DISCOUNT, payload(case(999_999, 10)))
    assert "999999" in report.rejected[0].raw
    assert "10000" in report.rejected[0].reason
