# PROJECT_CONTEXT — RippleTest

## One-liner

**RippleTest — Don't just find what changed. Discover what it might break.**

An inspectable, AI-assisted *differential testing* tool. A coding-focused open-weight model proposes interesting inputs for two versions of a Python function. A deterministic Python engine then executes the **same inputs** on both versions and shows concrete evidence of changed behaviour.

## Problem

A small code edit may alter results at important boundary cases. Developers may have passing tests that do not sample the inputs exposing the change. Conventional diff views explain lines changed, not the behavioural consequence. We want rapid, reproducible **counterexamples**: inputs where old and new implementations produce different outcomes.

## Target user

- Primary: developer reviewing a small function change.
- Secondary: hackathon judge seeing a transparent, self-contained tool.

## Main user story

“As a developer, I select a known code-change scenario, use a local open-weight model to propose relevant inputs, execute those inputs against both versions, and inspect the exact evidence of behavioural differences.”

## What AI does / does not do

**AI does:** inspect two snippets and their declared intended contract; propose 5–8 structured test inputs; provide brief reasoning for why each might matter.

**Python does:** validate args, invoke trusted function versions, compare actual outputs/exceptions, count outcomes, record reproducible evidence. The model must not fabricate results.

## Key innovation / honest framing

This is **not** the invention of regression testing, differential testing, or AI-assisted test generation. The hackathon contribution is a compact, open-weight, human-auditable workflow connecting AI-assisted case discovery to *actual before/after execution evidence*. In contrast to a code-dependency graph alone, the system actively searches for test inputs that reveal changed behaviour.

## Distinction from Graphify

Graphify reportedly supports code relationships and change impact. RippleTest focuses on **execution-based behavioural evidence**. Do not promise a live Graphify integration or specific submission steps without event instructions.

## Success moment

In the shipping threshold scenario, the model proposes a case where `cart_total == threshold`, and the deterministic engine shows that the old version returns `0` shipping while the modified version returns `50`.

## Truthful boundaries

- Finding different outputs demonstrates changed behaviour, *not* necessarily unintended behaviour.
- Finding no difference does not prove absence of regressions.
- This MVP tests selected, trusted examples only. It does not scan arbitrary repos or safely execute untrusted code.
- The model may be imperfect. Failed AI responses are visible rather than covered up.

## Product voice

Precise, calm, technical, and encouraging. Prefer “Behaviour changed” or “Needs review” over sensational claims like “Critical bug detected!”; prefer evidence over AI confidence percentages.
