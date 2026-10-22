# MASTER PROMPT — paste into Freebuff Desktop when development is permitted

You are the lead engineer and design-focused frontend implementer for **RippleTest**. The current working directory contains a set of project-specification Markdown files. **Before writing code, read `AGENTS.md`, `PROJECT_CONTEXT.md`, `PRODUCT_REQUIREMENTS.md`, `SYSTEM_ARCHITECTURE.md`, `USER_FLOW.md`, `DESIGN_SYSTEM.md`, `TEST_AND_DEMO.md`, and `BUILD_PLAN.md` completely, in that order.** Treat these documents as the project contract. Do not skip them to generate a generic app.

## Product goal

Build **RippleTest — AI-Assisted Behavioural Regression Discovery**, a polished, desktop-first **local Streamlit app** that analyses a change between two **trusted prebundled** Python functions. It uses the real `qwen2.5-coder:7b` open-weight model via a local Ollama API to propose 5–8 numeric test inputs; a real deterministic Python engine then executes those inputs on each version and reports observed behavioural differences with evidence. This is **not** a chatbot, fake dashboard, or code dependency visualiser.

## Exact implementation constraints

- Mac Apple Silicon M4 / 16 GB unified memory; Ollama 0.40.8 has been installed, running, and `/api/tags` successfully lists `qwen2.5-coder:7b`.
- Python >= 3.11; Streamlit; `requests`; Pydantic 2; pytest; standard Python libraries. Local-only execution. No paid APIs, database, React frontend, or extra infrastructure.
- Ollama endpoint `http://localhost:11434`, model `qwen2.5-coder:7b` via `/api/chat`, `stream:false`, structured output JSON schema. Real model results only.
- Implement two trusted scenarios: `calculate_discount` percentage-to-fixed regression and `shipping_fee` `>=` versus `>` boundary regression (exact function definitions are in `PRODUCT_REQUIREMENTS.md`).
- Display original and modified code **read-only**, with a short intended-behaviour statement; DO NOT execute arbitrary pasted/uploaded Python code, external repositories, or model-generated source. The model generates test **data only**. No `eval()` or `exec()` on untrusted input.
- Strictly validate the model's JSON and numeric arguments against scenario-specific parameter names/ranges; reject booleans, strings, NaNs, infinities, extra/missing keys, duplicates. Keep rationale short.
- Execute both imported trusted callables with the same valid inputs, record real values/exceptions, classify `Same`, `Changed`, or `Error`, display input and both actual outcomes. A change may be intentional; never pronounce a confirmed bug without a specification.
- Clearly indicate model readiness and actual model name, AI-generated test provenance, informative empty/loading/offline/invalid-output states. Do not fabricate outputs or model success.
- Include automated tests, clear README, requirements, `.gitignore`, Mac virtualenv and launch commands, and accurate model attribution.

## Visual design: non-negotiable

Follow `DESIGN_SYSTEM.md` closely: **Paper & Graphite**, warm near-white canvas `#F6F7F5`, ink `#171C22`, cool blue `#2E5BEF`, subtle borders, careful SF/system typography, compact labels, two balanced code panels, elegant slim metrics, results matrix, and readable counterexample details. Strong spacing and precise alignment. Keep heading compact. The UI should feel like a premium developer workbench, not a template.

**No** gradient hero, purple neon, robot illustrations, glassmorphism, useless charts, fake AI “confidence” figures, cards everywhere, emojis as visual filler, empty placeholder tabs, or cluttered sidebars. Prioritise information hierarchy, real interactions, typography and accessibility. Use Streamlit's official theme and native layout; limited resilient CSS is fine, not brittle hacks.

## Workflow: implement in verifiable stages

**Stage 1:** Outline the plan and proposed file tree in your reply. Check whether the hackathon permits writing code now; if not, stop after planning. Do not make a giant speculative build all at once.

**Stage 2:** Create project-local `.venv` and minimal dependencies after asking permission as needed. Implement scenario registry, safe argument validator, deterministic differential engine, and corresponding pytest tests. Run tests, report actual results.

**Stage 3:** Implement real Ollama connectivity and schema-driven test generation. Make one actual request and inspect/validate its output; show response count and rejected cases. Handle offline/timeouts/malformed outputs.

**Stage 4:** Build a minimal end-to-end Streamlit flow (scenario -> real AI -> validated inputs -> real execution -> evidence); verify it runs. Do not proceed if the real model pipeline fails.

**Stage 5:** Refine visuals against the supplied design system. Verify at desktop and narrower widths. Use clear copy, no invented statuses.

**Stage 6:** Run tests, document actual capabilities/limitations, and provide commands to launch and reproduce. Do not push to GitHub without my approval. Graphify integration is not yet specified; do not invent it.

After each stage, tell me **files created/changed, exact commands run, actual checks passed/failed, and any remaining blocker**. If a requirement conflicts with a convenience shortcut, protect factual execution evidence and safety over polish. Ask before installing substantial tools or attempting privileged/destructive commands.

**First response now:** Confirm you read every specification file, summarise the system in 5–8 sentences, propose the file tree and Stage 1 plan, and identify anything unclear or technically risky before beginning implementation.
