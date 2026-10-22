# TEST_AND_DEMO — empirical criteria and judging story

## Unit-level truth table

### A. Percentage discount

Original: `price - price * discount / 100`  
Modified: `price - discount`

| Input (`price`, `discount`) | Original | Modified | Difference? |
|---|---:|---:|---|
| (100, 10) | 90 | 90 | No |
| (1000, 10) | 900 | 990 | Yes |
| (0, 5) | 0 | -5 | Yes |
| (100, 100) | 0 | 0 | No |
| (250, 25) | 187.5 | 225 | Yes |

### B. Shipping threshold

Original: `0 if cart_total >= threshold else 50`  
Modified: `0 if cart_total > threshold else 50`

| Input (`cart_total`, `threshold`) | Original | Modified | Difference? |
|---|---:|---:|---|
| (400, 500) | 50 | 50 | No |
| (500, 500) | 0 | 50 | Yes |
| (600, 500) | 0 | 0 | No |
| (0, 0) | 0 | 50 | Yes |

These inputs are **expected assertions for development and fallback baseline tests**. They are **not** evidence that the AI model will generate the same cases in every run.

## Required automated tests

- Scenario registry selects correct trusted callables and source text is consistent with each callable.
- Differential engine correctly marks same vs changed for numeric outputs.
- Repeated test execution is deterministic for bundled pure functions.
- Comparator captures an exception as evidence without taking down the app.
- Validator rejects strings, booleans, non-finite numbers, extra/missing keys, out-of-range values, duplicate cases.
- Validation and UI display preserve the original model-provided rationale as **a suggestion**, not as the execution evidence.
- Ollama errors: request timeout, service unavailable, model missing, malformed JSON, insufficient valid cases.
- Scenario switch clears prior outcomes.
- If reporting implemented, summary counts match records and model provenance is accurate.

## Manual acceptance tests

1. `ollama list` shows `qwen2.5-coder:7b`.
2. With Ollama running, app shows connected model status.
3. Press Generate & run on discount scenario and inspect real proposed cases.
4. Confirm table outputs are real function results; independently reproduce at least one changed case with Python.
5. Switch to shipping, regenerate, and see whether model finds equality boundary; if not, mark honestly and discuss limitations.
6. Quit Ollama and retry: helpful error, no fabricated success.
7. UI at ~1440 x 900: readable, uncluttered, no generic template gradients.
8. README launch works using a fresh venv and locally installed model.

## Judge demo outline (approximately 75–90 seconds; adapt to actual rule)

- **0–15s Problem:** “A two-character code edit can silently change boundary behaviour even when ordinary tests pass.”
- **15–30s Distinction:** “Dependency maps tell us where code connects. RippleTest searches for input-level behaviour differences.”
- **30–60s Live:** select `Shipping threshold`, show `>=` versus `>`, click Generate & run; display exact input/output evidence if model produces boundary case. If not, use honest baseline test labelled curated and explain why real AI test discovery is probabilistic.
- **60–75s Architecture:** “Qwen2.5-Coder 7B runs locally with Ollama; it proposes *data*. Python validates and executes trusted code. No model-generated code is run.”
- **75–90s Limitation / next steps:** “Changed outcomes require review; our MVP uses bundled scenarios; production expansion would require secure code sandboxing and repository integration.”

## Questions judges may ask

**Why AI?** It proposes context-sensitive candidate inputs from source and intent; the deterministic engine verifies outcomes. Contrast this with a fixed test-only comparator: that would need a human to supply every input.

**Why open-weight?** Qwen2.5-Coder 7B runs locally via Ollama; include precise attribution and license in README. No proprietary model required at runtime.

**Are the results guaranteed accurate?** The recorded outputs are from actual trusted Python execution; classification detects differences. It does not independently establish correctness or ensure comprehensive test coverage.

**How novel?** The contribution is the understandable, reproducible integration of AI-assisted input generation with execution evidence, not a claim to have invented differential testing.

**Can users upload arbitrary code?** Not in the MVP because secure execution is nontrivial. Only vetted built-in scenarios are executed.

## Submission integrity

- Never stage `.venv`, model weight files, `.env`, tokens, proprietary code.
- Don't claim support for GitHub/Graphify integration until implementation and actual organiser instructions are verified.
- Include model attribution: Qwen2.5-Coder 7B; check upstream and variant license and list it accurately.
- Demo screenshots/videos, if used, must reflect a real observed run and be labelled with date/scenario; not AI-rendered mock results.
