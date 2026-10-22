# PRODUCT_REQUIREMENTS — scope and acceptance criteria

## MVP priority P0 (every item required)

1. **One-page working local Streamlit app**; wide desktop layout, no authentication/database.
2. **Two bundled scenarios**, selected with a compact control. Both have original and modified Python functions, a business-intent note, and input constraints. Code is displayed **read-only** in two side-by-side panes.
3. **Real local Ollama connection**, model `qwen2.5-coder:7b`. Visible connection state: checking, available, unavailable. A missing model must show actionable help.
4. **Generate & run tests** button: request 5–8 candidate test cases from Ollama, parse as strict JSON via schema/validation, reject unsafe/incompatible entries, execute accepted test cases against both trusted functions.
5. **Deterministic differential comparator:** record input, old output/new output or exception type, classification (`same`, `changed`, `error`), and model reason. The UI never invents outcomes.
6. **Evidence-first result view:** count total executed, changed, same, error; sortable/readable results list; select a row to see input and both observed results with the reason it was proposed. Do not claim a changed result is an actual bug.
7. **Error states:** Ollama offline, inference timeout, malformed JSON, zero usable cases, per-case execution failure. No fake demo data labelled live.
8. **Documentation and tests:** `.venv` instructions, reproducible `requirements.txt`, clear README, pytest covering comparator and scenarios.

## P1 nice-to-have (only after all P0s work)

- A useful changed-line visualisation (`difflib`), with accessible colour contrast.
- An explicit **Run curated baseline cases** button that works without the model, clearly labelled as a separate non-AI fallback demo.
- Export a JSON or Markdown report with actual results and model metadata.
- A second test round / “Generate more candidates” button and deduplicated test cases.
- Keyboard and narrow-width refinements.

## Out of scope

- User-edited/pasted arbitrary code **execution**, user uploads, repository scanning, GitHub pull requests, integration with real CI pipelines.
- External hosted models, paid APIs, claims of absolute correctness, training/fine-tuning.
- Multi-agent swarms, autonomous patching, full static analysis, production sandboxing.
- Pretend Graphify integration, speculative metrics (“reliability 97%”), fabricated testimonials.

## P0 demo scenarios (read-only)

### Scenario A — Percentage discount semantics

**Intent:** `discount` means a percentage from 0 to 100.

Original:
```python
def calculate_discount(price, discount):
    return price - (price * discount / 100)
```
Modified:
```python
def calculate_discount(price, discount):
    return price - discount
```
Parameters: `price` [0, 10000]; `discount` [0, 100]; finite numbers, not booleans.

### Scenario B — Free shipping boundary

**Intent:** shipping is free when cart total is **at least** the threshold; otherwise fixed fee 50.

Original:
```python
def shipping_fee(cart_total, threshold):
    return 0 if cart_total >= threshold else 50
```
Modified:
```python
def shipping_fee(cart_total, threshold):
    return 0 if cart_total > threshold else 50
```
Parameters: `cart_total` [0, 10000]; `threshold` [0, 10000]; finite numbers, not booleans.

These are **trusted function definitions authored in the application's own module**, not `exec()` strings dynamically compiled from a textbox. The UI may show the source via `inspect.getsource` or other read-only text.

## User acceptance criteria

- Launch app, select either scenario, view both versions and the intended contract.
- Ollama connected indicator reflects reality and exact model name.
- Press Generate & run; a genuine model request is made; returned structured input list is validated and inspected.
- Tests execute; results are based only on function calls; evidence can be manually reproduced.
- For discount, a case `price=1000, discount=10` yields `900` versus `990`. For shipping, `cart_total=500, threshold=500` yields `0` versus `50`.
- Unplug/quit Ollama and retry: the UI gives an error and never displays bogus new results.
- A README enables judges to reproduce the demo on a machine with requirements and model installed.
