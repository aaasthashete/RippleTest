# RippleTest — context pack (read this first)

**Status:** Product and engineering specifications only. This bundle does **not** contain a functioning application or prewritten implementation code.

## Purpose
Provide a single source of truth for a coding agent building **RippleTest — AI-Assisted Behavioural Regression Discovery** at an authorised hackathon.

## Exact files in this pack

1. `AGENTS.md` — non-negotiable instructions for Freebuff or another coding agent.
2. `PROJECT_CONTEXT.md` — user problem, positioning, novelty, and scope.
3. `PRODUCT_REQUIREMENTS.md` — requirements, exclusions, success criteria.
4. `SYSTEM_ARCHITECTURE.md` — stack, components, model calls, validation, testing boundary.
5. `USER_FLOW.md` — screen content, steps, states, and microcopy.
6. `DESIGN_SYSTEM.md` — detailed UI/UX specification and design rules.
7. `BUILD_PLAN.md` — time-boxed execution sequence, risks, fallback plan.
8. `TEST_AND_DEMO.md` — realistic example scenarios, assertions, demo script.
9. `MASTER_PROMPT.md` — kickoff instructions to paste into the coding agent.

## Setup with Freebuff Desktop on macOS

1. Unzip this pack.
2. Put **all the `.md` files directly inside your existing `~/Documents/RippleTest/` folder** (not nested in another subfolder). If your project is somewhere else, put them in that project's root instead.
3. Open that folder in Freebuff Desktop.
4. Check the event policy: if work cannot be done before the event starts, keep this as a preparation specification and only ask the agent to write code at the permitted time.
5. Paste the contents of `MASTER_PROMPT.md` into the Freebuff conversation, or simply tell Freebuff to read that file and execute its instructions. Review every proposed shell command or install.
6. **Don't allow Freebuff to skip actual tests, substitute a proprietary cloud model, or display fabricated demo results.**

## Verified hardware/environment from setup

- Apple Silicon Mac M4, approximately 16 GB unified memory.
- Ollama `0.40.8` running on the Mac.
- `qwen2.5-coder:7b` downloaded and available via local Ollama.
- `curl http://localhost:11434/api/tags` succeeds and lists that model.
- The model already returned useful discount-test inputs in a manual chat experiment, though the chat answer came in Markdown fences. The app must request and validate structured JSON.
- Freebuff Desktop installed. Mac-local application; no public URL is required for the core demo.

## Important caveats

- Do not claim an untested model, feature, or deployment works.
- A **behaviour difference** is not necessarily a bug. Avoid the words “confirmed regression” absent an independent specification/oracle.
- This MVP executes only trusted, developer-bundled Python functions. It must not execute arbitrary user- or model-provided source code.
- Graphify instructions from the organiser have **not yet been verified**. Keep the project Git-friendly, but do not invent any Graphify integration.
