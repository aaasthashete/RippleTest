# RippleTest

### Find the break before it ships.

**AI-assisted behavioural regression discovery · Local, open-weight AI · Evidence-first testing**

A single line of code can change behaviour far beyond the case a developer remembered to test. **RippleTest** compares two versions of a Python function, asks a locally running coding model to suggest targeted inputs, and executes those inputs against **both versions** to reveal reproducible behavioural differences.

**The model suggests the tests. The Python engine produces the evidence.**

> Built for the **Best Open-Source AI Project** hackathon track. RippleTest is a functional prototype focused on small numerical Python functions—not a general-purpose repository scanner or a replacement for an existing test suite.

## The problem

Existing tests can miss the particular inputs that expose a regression: an equality boundary, a zero value, or a percentage interpreted incorrectly. Developers need a fast way to ask **“What changed in behaviour?”** and receive the exact input and outputs that demonstrate it.

## What RippleTest does

1. **Compare code:** Inspect or edit the original and modified versions of a supported Python function.
2. **Generate tests with open-weight AI:** Qwen2.5-Coder 7B proposes relevant numerical inputs through local Ollama.
3. **Validate inputs:** Reject malformed, incompatible, duplicate, non-finite, and out-of-range proposals.
4. **Execute both versions:** Run identical accepted inputs through the appropriate Python execution engine.
5. **Inspect evidence:** See matching results, behavioural differences, exceptions, model provenance, individual test rationale, and an exportable report.

Two bundled examples are available via **Load example**, and a clearly labelled **Run curated baseline** option works without AI when an unchanged built-in example is selected. Curated inputs are **never presented as AI-generated**.

## See it in action

### Example 1 — A discount that silently changes meaning

**Original** — `discount` represents a percentage:

```python
def calculate_discount(price, discount):
    return price - (price * discount / 100)
```

**Modified** — the same argument is subtracted as a fixed amount:

```python
def calculate_discount(price, discount):
    return price - discount
```

For `price=1000, discount=10`:

| Input | Original | Modified | Observation |
|---|---:|---:|---|
| `1000, 10` | `900` | `990` | Behaviour changed |
| `100, 10` | `90` | `90` | Same result |

### Example 2 — The free-shipping equality boundary

Changing `cart_total >= threshold` to `cart_total > threshold` makes a difference **exactly at the threshold**. For `cart_total=500, threshold=500`, the original charges `0`, while the modified version charges `50`.

These are **reproducible illustration inputs**, not a promise that the model will generate the same inputs on every run. The UI displays the inputs and results actually used in each run.

## How it works

```text
Original function ─┐
                   ├──► Python AST analysis / execution-target selection
Modified function ─┘                         │
                                            ▼
                        Local Ollama → Qwen2.5-Coder 7B
                                            │
                               Proposed JSON test inputs
                                            │
                              Strict input validation
                                            │
                               Differential execution
                               ┌──────────┴──────────┐
                               ▼                     ▼
                        Original version       Modified version
                               └──────────┬──────────┘
                                          ▼
                         Actual outcomes + exceptions
                                          ▼
                          Evidence table + JSON export
```

**Why use AI?** Writing a few fixed test cases is easy; choosing *useful* inputs for an unfamiliar code change is harder. The open-weight coding model proposes candidate edge cases, while deterministic code handles validation, execution, and comparison. RippleTest does **not** ask the model to invent expected outputs or declare a change correct.

### Two execution paths

- **Bundled examples:** Run the reviewed Python functions included in `core/scenarios.py`.
- **Edited functions:** Parse the actual editor contents using Python's `ast` module and evaluate only a restricted set of numerical expressions through `core/expression.py`. The app does **not** run pasted source with `eval()` or `exec()`.

The edited-code path supports one function with compatible numeric parameters and a single `return` expression; arithmetic, comparisons, and conditional expressions are supported. Imports, function calls, loops, assignments, object access, and arbitrary Python programs are **not supported**. Unsupported input receives a visible error instead of pretending it was analysed.

## Technology stack

| Component | Technology | Responsibility |
|---|---|---|
| Interface | **Streamlit** | Editable code workspace, analysis states, evidence and export |
| Runtime model | **Qwen2.5-Coder 7B** | Proposes test inputs and brief rationales |
| Local model server | **Ollama** | Serves the open-weight model over `localhost:11434` |
| Analysis & validation | **Python AST + Pydantic / project validators** | Restricts supported syntax and validates model proposals |
| Execution | **Python differential engine** | Runs both implementations and compares actual outcomes |
| Testing | **pytest** | Unit and app-state tests |

**Open-weight requirement:** The core AI capability uses `qwen2.5-coder:7b`, a Qwen2.5-Coder model served locally through Ollama. The 7B model is distributed under **Apache-2.0**; RippleTest does not train or redistribute model weights. The coding assistant used during development is separate from the model used by RippleTest itself.

## Run locally

**Requirements:** macOS or another Ollama-supported system, **Python 3.11+**, sufficient memory for Qwen2.5-Coder 7B, and Ollama installed. The prototype was developed on an **Apple M4 Mac with 16 GB unified memory**.

```bash
# 1. Clone the repository
git clone https://github.com/aaasthashete/RippleTest.git
cd RippleTest

# 2. Create and activate a Python environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
python -m pip install -r requirements.txt

# 4. Install the local AI model (with Ollama running)
ollama pull qwen2.5-coder:7b

# 5. Start the app
streamlit run app.py
```

Open the local address printed by Streamlit (usually **http://localhost:8501**).

If Ollama is not already running, start the Ollama desktop app or run `ollama serve` in a separate terminal. **No paid model API or API key is required.** Model generation requires the local Ollama service, but the curated baseline for unchanged bundled examples does not.

### Test the project

```bash
python -m pytest -q
```

## Project structure

```text
RippleTest/
├── app.py                     # Streamlit user interface
├── core/
│   ├── scenarios.py           # Reviewed demo functions and curated inputs
│   ├── expression.py          # Restricted AST interpreter for edited functions
│   ├── ollama_client.py       # Local model health checks and generation
│   ├── validators.py          # Model-response and input validation
│   ├── differential.py        # Execution and result classification
│   ├── pipeline.py            # Run orchestration and provenance
│   └── report.py              # Export of observed results
├── tests/                     # Automated tests
├── .streamlit/config.toml     # UI theme configuration
├── requirements.txt
└── README.md
```

## Reliability, boundaries, and limitations

- A **behavioural difference is not automatically a bug**; the developer must determine intended behaviour.
- **No difference detected** does not prove the code is regression-free. The model may miss useful inputs.
- AI generates **input data**, not executable Python or fabricated test outcomes. Failed validation is reported rather than silently ignored.
- The restricted AST interpreter is **not a general sandbox** and does not support arbitrary Python, external repositories, or executing uploaded files.
- This is a **local function-level MVP**: there is no full GitHub PR analysis, dependency graph, CI/CD integration, or cloud-hosted inference in the application.
- The interface reports model connectivity and distinguishes AI-generated cases from the curated baseline.

## Future direction

RippleTest could evolve into a change-focused regression testing service by adding repository dependency analysis, isolated parallel test runners, automated CI/CD workflows, broader language coverage, and stronger regression-oracle support. **These are future possibilities, not features claimed in this hackathon MVP.**

---

**RippleTest** — *Don't just find what changed. Discover what it might break.*
