# AGENTS.md — AI-agent project instructions

## Read before writing code

You are building **RippleTest**, a Mac-local, single-page Python developer tool for discovering changed behaviour across **trusted paired Python functions**. Read these files in order before implementation:

1. `PROJECT_CONTEXT.md`
2. `PRODUCT_REQUIREMENTS.md`
3. `SYSTEM_ARCHITECTURE.md`
4. `USER_FLOW.md`
5. `DESIGN_SYSTEM.md`
6. `TEST_AND_DEMO.md`
7. `BUILD_PLAN.md`

Read `MASTER_PROMPT.md` for build sequencing. If two documents differ, honour this file's safety restrictions first, then `PRODUCT_REQUIREMENTS.md`, then `SYSTEM_ARCHITECTURE.md`.

## Hard requirements

- Use **Python + Streamlit** for the app, **local Ollama** at `http://localhost:11434`, model **`qwen2.5-coder:7b`**, `requests`, and Pydantic for JSON validation.
- Build an **end-to-end vertical slice** first: trusted scenario -> real Ollama JSON request -> validated AI-proposed inputs -> execute both bundled functions -> compare real outputs -> show actual evidence.
- The model must be real, central, and disclosed in the UI and README. Do **not** silently switch to proprietary APIs or hardcoded “AI” results.
- The model generates **data only** (numeric function arguments and a brief rationale), never code to execute.
- Code visible in the UI is read-only in the MVP. Execute **only bundled, pre-reviewed Python functions imported from the application itself**. Never run user-pasted/uploaded files or model-generated code. No `eval`, `exec`, dynamic importing user paths, or `shell=True` with untrusted input.
- Distinguish: `Same outcome`, `Behaviour changed`, `Execution error`, `AI response invalid`. Do not label differences “bugs” or “tests failed” automatically.
- Validate every input; handle Ollama disconnection, timeout, invalid JSON, incompatible arguments, and exceptions without fake success.
- Keep model-generated and prewritten baseline tests clearly labelled by provenance. In particular, never silently replace AI output with handcrafted cases.
- Write small, readable, beginner-explainable modules. Include comments where the logic is non-obvious.
- Use a `.venv`, `requirements.txt`, `.gitignore`, `.streamlit/config.toml`, README, and pytest tests.
- No database, authentication, React, Next.js, Docker, remote inference, paid APIs, or complex orchestration in the MVP.
- Ask before using destructive file operations, adding expensive services, creating a GitHub repository, or pushing commits. Do not use `sudo`.

## UI quality rules

- Implement the **Paper & Graphite** design in `DESIGN_SYSTEM.md` with Streamlit-compatible components; use restrained CSS and official theming, not a fragile framework rewrite.
- No purple gradient hero, glowing AI icons, oversized cards, emojis as decoration, invented statistics, empty charts, fake notifications, or meaningless progress meters.
- The results and the evidence matter more than decoration. Design should feel like a thoughtful developer product.

## Engineering quality rules

- On each milestone, run tests, report actual results, and note what remains unverified.
- Preserve meaningful, non-failing UI states across Streamlit reruns with `st.session_state`.
- Give a clear Mac launch command at the end; project must work with local Ollama and no cloud key.
- Use the exact acceptance checklist in `TEST_AND_DEMO.md`.
- If an implementation requires relaxing a restriction, **stop and ask**, rather than silently weakening it.

## Hackathon integrity

These documents are pre-event planning material. Before implementation, obey the event's rules on AI assistance, prebuilt source code, templates, team work, and submission timing. If the organiser disallows prebuilt code, do not generate code until the authorised window.
