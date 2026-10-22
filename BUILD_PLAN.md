# BUILD_PLAN — 3–4 hour hackathon execution

**Precondition:** Confirm official rules before coding; this pack is planning material and must not be mistaken for a prebuilt submission. Adapt to the actual competition timing and available team members.

## Sequence / gatekeeping

### Gate 0 — Environment (pre-event where permitted)
- Mac M4 16 GB; Ollama installed and available; Qwen2.5-Coder 7B downloaded.
- Freebuff Desktop project folder created.
- Verify Python 3.11+, pip/venv, free disk, Streamlit package install permission.
- Confirm GitHub login and organiser rules / Graphify instructions without inventing integration steps.

### Phase 1 — Engine first (0:00–0:55)
- Create venv, requirements, trusted demo registry and test-comparison engine.
- Verify explicit known results for 2 scenarios with **curated unit-test inputs** (not pretending model generated them).
- Ensure no arbitrary code execution path exists.

### Phase 2 — Real AI (0:55–1:45)
- Add Ollama status + real POST `/api/chat` with schema.
- Validate generated inputs; run through differential comparator.
- Test with real 7B model, retry prompt tuning if necessary. Don't hide poor outputs.

### Phase 3 — Premium Streamlit UI (1:45–2:40)
- Implement Paper & Graphite design, scenario switch, read-only code, CTA, result matrix, row inspector, empty/ready/loading/error states.
- Verify layout on Mac browser at realistic demo window size.

### Phase 4 — Hardening and demo (2:40–3:20)
- Add pytest tests, source/implementation consistency test, offline/invalid JSON scenarios.
- Demo both discount and threshold regressions, record actual counts and failures.
- Improve accuracy, reliability, keyboard/accessibility before decorative polish.

### Phase 5 — Submission (3:20–4:00)
- README: prerequisites, install, run, model attribution/license, security limitations, actual features.
- `.gitignore` includes `.venv`, `.DS_Store`, caches, secrets and huge model files.
- Git commit + push **when approved by user**; follow organiser Graphify instructions once obtained.
- Judge demo rehearsal and backup: a clear record of a previous genuine run, separately labelled and not substituted as live.

## If only 3 hours

Cut optional diff line colouring, baseline button, report export, second-pass AI generation, advanced animations. Maintain the real Ollama call, validated inputs, actual execution, honest results, and clear UI.

## Top implementation risks

1. Model outputs unsuitable JSON -> schema + strict validator + clear retry error.
2. 7B generation too slow -> keep short 4k-token context; test speed; optionally benchmark smaller model **with explicit label**, never switch silently.
3. Strange Streamlit layout -> prioritise native components and official theme configuration.
4. Offline Ollama -> actionable status instead of success; optional separate curated baseline demo.
5. App does not run on judge machine -> demo on your Mac, include reproducible README; only promise hosted demo if actually deployed.
6. GitHub/Graphify submission unknown -> obtain event instructions, budget time; don't assume an automatic integration.

## Agent work protocol

Do not generate everything at once without verification. Follow these increments:

1. Create dependency and file skeleton; show planned file tree.
2. Implement + pytest differential engine; report real passing/failing commands.
3. Implement Ollama client; run a genuine model call and show actual parsed input count.
4. Connect minimal Streamlit UI; launch and visually review.
5. Refine design to specification and test both demos.
6. Document and prepare submission.

After every increment state: **files changed, commands run, test results, remaining blockers**.
