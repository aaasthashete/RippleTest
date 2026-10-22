"""One run of RippleTest: validated inputs in, real evidence out."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from .differential import CaseResult, Provenance, Summary, run_cases, summarise
from .scenarios import Scenario
from .validators import ValidationReport, ValidatedCase

from .expression import ENGINE_LABELS, ENGINE_TRUSTED, engine_for

if TYPE_CHECKING:
    from .expression import FunctionPair


@dataclass(frozen=True)
class RunRecord:
    """Everything needed to render and export one comparison run."""

    run_id: str
    scenario_id: str
    scenario_label: str
    function_name: str
    provenance: Provenance
    results: tuple[CaseResult, ...]
    validation: ValidationReport
    created_at: str
    model: str | None = None
    duration_seconds: float | None = None
    engine: str = ENGINE_TRUSTED

    @property
    def summary(self) -> Summary:
        return summarise(self.results)

    @property
    def engine_label(self) -> str:
        """Which execution engine actually produced these outcomes."""
        return ENGINE_LABELS.get(self.engine, self.engine)


def build_run(
    scenario: "Scenario | FunctionPair",
    validation: ValidationReport,
    provenance: Provenance,
    *,
    model: str | None = None,
    duration_seconds: float | None = None,
) -> RunRecord:
    if not validation.accepted:
        raise ValueError("No accepted cases to execute.")

    results = run_cases(scenario, validation.accepted, provenance)
    return RunRecord(
        run_id=uuid4().hex[:12],
        scenario_id=scenario.id,
        scenario_label=scenario.label,
        function_name=scenario.function_name,
        provenance=provenance,
        results=tuple(results),
        validation=validation,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=model,
        duration_seconds=duration_seconds,
        engine=engine_for(scenario),
    )


def run_curated_baseline(scenario: Scenario) -> RunRecord:
    """Execute the pre-written baseline cases, explicitly not AI-generated."""
    cases = tuple(
        ValidatedCase(
            number=number,
            arguments=dict(case.arguments),
            reason=case.note,
        )
        for number, case in enumerate(scenario.curated_cases, start=1)
    )
    validation = ValidationReport(accepted=cases, rejected=())
    return build_run(scenario, validation, provenance="curated")
