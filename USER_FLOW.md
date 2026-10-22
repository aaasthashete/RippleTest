# USER_FLOW — interaction and information architecture

## Single-screen workflow

**Open app -> select trusted scenario -> read original/modified code and intended behaviour -> check model ready -> click Generate & run tests -> view observed outcomes -> inspect counterexamples -> optionally export / repeat.**

No setup wizard, login, or extra pages for MVP.

## Screen structure (desktop-first)

```
[ R/ RippleTest ]  [DIFFERENTIAL TESTING WORKSPACE]        [Local model • Ready]

Understand what your change actually does.
Small explainer line.                      [Scenario ▾] [Generate & run tests]

[ What this function should do — brief business contract ]

[ ORIGINAL · v1  ]                     [ MODIFIED · v2 ]
[ read-only Python snippet ]           [ read-only Python snippet ]
[ line numbers / subtle diff indicators ]

[ Tested 06 ] [ Same 04 ] [ Changed 02 ] [ Errors 00 ]

[ TEST MATRIX / outcomes                            ]
[ input summary   old result    new result    status ]
[ 1000, 10        900           990           Changed ]
[ ...                                              ]

[ Selected test explanation: inputs, real observed outputs, AI reason ]
[ Testing scope note: behavioural difference != confirmed bug     ]
```

On a narrower screen, stack code panels and results sections; no horizontal clipping of key controls.

## Detailed interactions

### First load

- One clear primary heading and concise subheading.
- Default scenario `Percentage discount`, no fabricated results.
- Original and modified code both readable without interaction.
- Model status shows `Checking local model…`, `Qwen2.5-Coder 7B · Ready`, or `Ollama unavailable`.
- Primary button `Generate & run tests`.
- Before tests: empty state “No tests run yet. Generate candidate inputs to compare both versions.”

### Scenario switch

- Update code, contract and constraints immediately.
- Clear any prior run or clearly label it as belonging to the earlier scenario; simplest is clearing.

### Generating

- Button loading state `Generating test inputs…`.
- Progress text reflects real stages: `Requesting local model`, `Validating cases`, `Executing trusted functions`, `Comparing outputs`.
- Never use invented time estimates or fake animated percentage completion.

### Valid model response

- Show model-proposed cases and test outcomes, with counts calculated from executed records.
- All summaries derived from programmatic comparator, never LLM text.
- Each row shows input args, original output, modified output, and status.
- Selecting/expanding a row shows rationale and complete inputs/outputs.
- Highlight changed rows; make matching tests less visually emphatic.

### No revealed difference

- Neutral message: “No behavioural differences observed in these N executed cases.”
- Supporting explanation: “This does not establish that the change is regression-free.”

### Ollama unavailable

- Actionable state: “Ollama is not reachable at localhost:11434. Open Ollama and ensure `qwen2.5-coder:7b` is installed.”
- Do not show old run as if it is fresh or swap to a hosted model.

### Invalid JSON / zero valid cases

- Message: “The local model didn't return usable test inputs. Try generating again.”
- Optional technical detail only if useful; never show fake generated test cases.

### Error during execution

- Show the input and exception details; preserve other valid results.
- Avoid claiming an exception equals a confirmed regression.

### Optional curated baseline tests

- If implemented, a separate secondary `Run baseline examples` action generates deterministic predefined cases and labels every record `Curated baseline`, not `AI-generated`.

## Microcopy rules

- Say “Behaviour changed”, not “BUG FOUND” by default.
- Say “Original” and “Modified”, never misleading “Expected” and “Actual” without a requirements oracle.
- Say “Cases executed”, not “Coverage %”; we do not calculate coverage in MVP.
- Explain local model: “Runs on this Mac through Ollama; no model API key required.”
- Avoid decorative toast spam, fake activity feeds, AI personality theatrics, or synthetic success metrics.
