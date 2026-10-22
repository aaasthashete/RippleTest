# SYSTEM_ARCHITECTURE — implementation contract

## Stack

- Python >= 3.11, local venv.
- Streamlit single-page UI (`app.py`).
- Ollama local HTTP API: default `http://localhost:11434`, fixed model `qwen2.5-coder:7b`.
- `requests` for HTTP; `pydantic>=2` for validated structured outputs; `pytest` for tests.
- Python standard library: `json`, `ast` / `difflib` (read-only inspection), `math`, `dataclasses` or typed models; `inspect` as needed.
- No external API keys, JS framework, background jobs, auth, or database.

## Suggested small file tree (to be GENERATED during the permitted build window)

```text
RippleTest/
├── app.py
├── core/
│   ├── __init__.py
│   ├── scenarios.py
│   ├── ollama_client.py
│   ├── validators.py
│   ├── differential.py
│   └── report.py                 # optional
├── tests/
│   ├── test_scenarios.py
│   ├── test_validation.py
│   └── test_differential.py
├── .streamlit/config.toml
├── requirements.txt
├── .gitignore
├── README.md
└── [these planning .md files]
```

## Data flow

1. UI selects a developer-defined scenario ID (`discount` or `shipping`).
2. Scenario registry returns exact functions, read-only source text, intended contract, parameter names/types, and argument bounds.
3. UI queries `GET /api/tags` for connection/model availability.
4. UI sends model two function snippets + intent + parameter/constraint schema. Use `POST /api/chat` with `stream:false`, an appropriate JSON Schema in `format`, `options.temperature` low (e.g. 0.1), and manageable context (e.g. `num_ctx:4096`).
5. Parse `message.content` as JSON. Validate that structure contains `test_cases` with 5–8 proposals, each has numeric named arguments and rationale. Note: some model outputs may fail; display an honest retry option, do not invent successful AI output.
6. Reject booleans, strings as numbers, `NaN`, infinities, missing/extra keys, out-of-range values, and duplicates. Use `math.isfinite`. Keep a short, human-readable validation message when rejecting entries.
7. Execute only registered, trusted original/modified callables with the **same** vetted named arguments, bounded inputs, no model-written code.
8. Capture output or exception for each. Compare finite numeric outputs with a small documented tolerance (e.g. `math.isclose` with `rel_tol=1e-9`, `abs_tol=1e-9`), compare nonnumeric results with equality, and compare exceptions explicitly. Label exception-related outcomes `error` for review rather than silently counting as a pass.
9. Return a typed result with evidence fields for rendering; summary is derived from actual result records.

## Expected model-response contract (conceptual)

```json
{
  "test_cases": [
    {
      "arguments": {"price": 1000, "discount": 10},
      "reason": "Checks a nonzero percentage on a larger price"
    }
  ]
}
```

Use a schema compatible with Ollama's structured `format` capability. Define argument field keys precisely based on selected scenario; not a permissive arbitrary dictionary alone. Avoid adding inferred expected outputs from the model. The engine computes real outcomes. If Pydantic version or schema proves incompatible with Ollama, simplify the schema but keep server-side validation.

## Trusted scenario registry

Expose a small immutable mapping such as:
- `scenario.id`, `.label`, `.contract`, `.original_source`, `.modified_source` (read-only display).
- `.original_callable`, `.modified_callable` (trusted imports, never created from text supplied to the app).
- `.parameters` and `.bounds`.

The source text and imported callables must correspond. Include unit tests to prevent visual source/implementation drift.

## Ollama connectivity

- Health check: `GET http://localhost:11434/api/tags`, timeout ~3 seconds.
- Generation: `POST http://localhost:11434/api/chat`, non-streaming, timeout ~90 seconds. Treat timeouts as recoverable errors; do not promise exact response speed.
- Explicit headers/JSON body; show model name next to readiness indicator.
- No `.env` with secrets needed for local Ollama. Keep URL and model string in one config location.

## UI state handling

Store selected scenario, generation state, candidate cases, validation issues, last result records, and source/model metadata in `st.session_state`. Clearing or switching scenario must not leave stale results from a previous scenario. During model generation disable duplicate clicks where practical.

## Threat model (limited MVP)

- The LLM is untrusted as an executable code author; it may only provide test **data**.
- Parsed JSON itself is untrusted. Strongly validate before calling trusted functions.
- **Never** run arbitrary Python strings using `exec`, `eval`, `ast.literal_eval` of unbounded structures, or a shell.
- Do not expose user-supplied custom Python editor whose content is executed: source panels are read-only.
- If a later version supports untrusted code, it requires a real isolated execution environment and a separate security design; out of current MVP.

## Diagnostics / transparency

Log error categories locally with no credentials. Provide useful UX errors rather than Python stack traces for the judge. Keep one-click details for advanced viewers if needed. Display whether a result is `AI-generated` or `curated baseline`. Do not show fictional “AI confidence” scores.
