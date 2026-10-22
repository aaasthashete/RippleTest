# RippleTest

**Don't just find what changed. Discover what it might break.**

RippleTest is a local, single-page developer workbench for **behavioural regression
discovery**. It takes two versions of a trusted Python function, asks a local
open-weight model to propose interesting numeric inputs, validates those inputs,
executes **both versions with the same arguments**, and shows the real
before/after evidence — including the exact counterexample when behaviour
changes.

The model is **real and central**: [`qwen2.5-coder:7b`](https://ollama.com/library/qwen2.5-coder)
served by [Ollama](https://ollama.com) at `http://localhost:11434`. There is no
hosted fallback and no API key. Every result displayed in the app comes from an
actual Python call; the model never supplies code or expected outputs.

---

## The 30-second demo

1. Select **Free-shipping boundary**. The panes show `>=` (original) and `>`
   (modified).
2. Click **Generate & run tests**.
3. Qwen2.5-Coder 7B proposes inputs such as `cart_total=500, threshold=500`.
4. Python calls both functions. The engine reports original `0`, modified `50`,
   outcome **Changed** — a reproducible counterexample at the equality boundary.

The same flow works for **Percentage discount** (`discount` as a percentage
versus a flat subtraction): `price=1000, discount=10` returns `900.0` in the
original and `990` in the modified version.

An offline **curated baseline** is available too. It is pre-written in the
application and every row is labelled `Curated baseline` / `not AI-generated`,
so it can never be mistaken for model output.

---

## Requirements

- macOS (developed on an Apple Silicon M4, 16 GB unified memory)
- Python **3.11+**
- [Ollama](https://ollama.com/download) running locally
- The model:

```bash
ollama pull qwen2.5-coder:7b
ollama list          # qwen2.5-coder:7b should appear
```

No cloud account, API key, database, or network access beyond the local Ollama
port is used.

## Setup and launch (macOS)

```bash
cd /path/to/RippleTest

# 1. Create the local virtual environment
python3.11 -m venv .venv

# 2. Install dependencies
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 3. Make sure Ollama is running and the model is present
ollama serve            # if it is not already running as a menu-bar app
ollama list

# 4. Launch the app
.venv/bin/streamlit run app.py
```

Streamlit opens `http://localhost:8501`. The header first shows the honest
`CHECKING` state, then the result of a real `GET /api/tags`
(`LOCAL / QWEN2.5-CODER 7B · READY`, a missing-model message, or `OLLAMA /
OFFLINE`); it is never hardcoded.

Verified locally with Python 3.11.15, Streamlit 1.58.0, and Ollama 0.40.8 serving
`qwen2.5-coder:7b` (Q4_K_M).

### Troubleshooting

| Symptom | Fix |
|---|---|
| `Ollama unavailable` | Start Ollama (`ollama serve` or the Mac app), then press **Check connection**. |
| `model missing` | `ollama pull qwen2.5-coder:7b` |
| First generation is slow | The 7B model may need to load into memory; later requests are faster. The app waits up to 120 s and reports a real timeout instead of inventing results. |
| Model returns unusable JSON | The app says so, shows the raw reply, rejects the proposals, and executes nothing. Press **Generate & run tests** to retry. |

---

## How it works

```
Scenario selected (trusted, read-only source)
        │
        ▼
GET  localhost:11434/api/tags ────────────► header status (real)
        │
POST localhost:11434/api/chat  (JSON schema, stream:false, temperature 0.1)
        │   model returns {test_cases:[{arguments:{...}, reason:"..."}]}
        ▼
Strict validation  ── reject booleans, strings, NaN/inf, missing/extra keys,
        │             out-of-range values, duplicates, malformed items
        ▼
Real execution: original(**args) and modified(**args), exceptions captured
        │
        ▼
Deterministic comparison → Same result / Changed / Error
        │
        ▼
Evidence matrix, case inspector, AI rationale, JSON export
```

### Trust boundary

- The model generates **data only**: numeric arguments plus a short rationale.
- Only the two bundled, pre-reviewed functions in [`core/scenarios.py`](core/scenarios.py)
  are ever called. The UI's code panes are read-only; they are derived from
  `inspect.getsource` of those same callables (a test enforces that the displayed
  source matches the executed function).
- There is no `eval`, `exec`, dynamic import of user paths, shell execution, or
  upload/paste-and-run path in the MVP. A test scans the core package for those
  facilities.
- Model-supplied arguments are validated against the selected scenario's exact
  parameter names, numeric type, finiteness, inclusive bounds, and uniqueness
  before any call.

### Classification

| Label | Meaning |
|---|---|
| **Same result** | Both versions returned equal values (numbers within `rel_tol=1e-9`, `abs_tol=1e-9`). |
| **Changed** | Both executions completed and returned different values. |
| **Error** | At least one version raised an exception. Captured as evidence and shown with its type and message; never counted as a pass. |

A changed value is **not** labelled a bug: the tool has no independent oracle for
intent. `No difference found` likewise does not prove the change is safe.
Exceptions during execution do not take the app down, and other cases still run.

### What the model actually receives

Both function sources, the intended behaviour, and the accepted argument names
with ranges. It is explicitly told not to include expected outputs and to include
boundary/equality cases where relevant. It receives **no** test result feedback,
so it cannot fabricate an outcome that appears in the matrix.

---

## Bundled scenarios

| Scenario | Original | Modified | Contract |
|---|---|---|---|
| Percentage discount | `price - (price * discount / 100)` | `price - discount` | `discount` is a percentage from 0 to 100 |
| Free-shipping boundary | `0 if cart_total >= threshold else 50` | `0 if cart_total > threshold else 50` | Free when the total is **at least** the threshold |

Bounds: `price`/`cart_total`/`threshold` ∈ [0, 10000], `discount` ∈ [0, 100];
finite numbers, booleans rejected.

The curated baseline cases (and their expected real outputs) are the ones in the
project's `TEST_AND_DEMO.md`. They are assertions for development, **not**
evidence of what the model will propose on any given run.

---

## Project structure

```text
RippleTest/
├── app.py                  # single-page Streamlit UI (Paper & Graphite)
├── core/
│   ├── scenarios.py        # trusted functions + read-only scenario metadata
│   ├── ollama_client.py    # real /api/tags + /api/chat client, typed errors
│   ├── validators.py       # JSON envelope parsing + strict numeric validation
│   ├── differential.py     # execution, exception capture, comparison, summary
│   ├── pipeline.py         # one run: validation → execution → RunRecord
│   └── report.py           # JSON export of actual records
├── tests/                  # pytest: scenarios, validation, comparison, Ollama errors
├── .streamlit/config.toml  # official theme settings
├── requirements.txt
├── pytest.ini
└── *.md                    # the original hackathon specification pack
```

## Running the tests

```bash
.venv/bin/python -m pytest
```

The suite uses faked HTTP for Ollama error paths, so it needs neither a running
model nor network access. It also drives the actual Streamlit page headlessly
through `AppTest` to check the offline, missing-model, invalid-reply,
zero-usable-case, and curated-baseline states. It covers:

- the scenario registry, source/callable consistency, and documented truth tables;
- rejection of booleans, strings, NaN/inf, missing/extra arguments, out-of-range
  values, duplicates, malformed items, and more than eight proposals;
- same/changed/error classification, numeric tolerance, exception capture,
  deterministic repeat runs, provenance, summary counts, and report export;
- real-client behaviour for ready/missing model, offline, timeout, empty reply,
  HTTP errors, and the one-time plain-JSON fallback if a runtime rejects a schema.

## Design

The interface follows the **Paper & Graphite** design system: warm near-white
canvas, graphite ink, one blue accent, restrained semantic colours, macOS system
typography, a two-pane read-only code comparison with diff-marked lines, a slim
four-metric summary, an outcome matrix, and an evidence inspector. No gradients,
glassmorphism, invented confidence scores, decorative charts, or filler metrics.
Narrow layouts stack the panes and wrap controls without hiding evidence.

## Model attribution

- Runtime model: **Qwen2.5-Coder 7B** (`qwen2.5-coder:7b`, 7.6B parameters,
  Q4_K_M quantisation, served by Ollama 0.40.8 locally).
- Upstream: Qwen2.5-Coder by the Qwen team, Alibaba Cloud. The 7B model card
  lists **Apache-2.0**; check the model card for the exact variant you install.
- RippleTest does not train, fine-tune, or redistribute model weights.

## Limits and non-goals

- Only two bundled scenarios; no repository scanning, GitHub PRs, CI integration,
  user code execution, database, auth, or hosted model.
- No coverage metric or correctness oracle. A difference may be intentional.
- Not a sandbox. Executing arbitrary user code safely would require a separate
  security design and is explicitly out of scope for this MVP.
- Model output is probabilistic: a run may miss the interesting boundary. The
  honest fallback is the clearly-labelled curated baseline, plus a visible retry.

## Hackathon integrity

These Markdown files were pre-event planning material. The implementation above
was produced during the event with AI assistance, which the event explicitly
allows. No prebuilt implementation code from the planning pack was used; the pack
contains specifications only. No Graphify or GitHub integration is claimed.
