# DESIGN_SYSTEM — Paper & Graphite

## Art direction

**A premium code-review workbench, not an AI-generated SaaS landing page.**

References in spirit: calm editorial software, precise data-dense developer tools, strong typography, tactile but subtle surfaces. It should communicate trust, craft, and analytical clarity. The visual focal point is the **side-by-side code comparison and observed counterexample** — not a decorative hero or chatbot.

## Visual personality

- Restrained, architectural, precise, confident.
- Minimal but not sterile; deliberate use of spacing and type hierarchy.
- One crisp accent and measured semantic state colours.
- Quiet micro-interactions; avoid large motion.
- Designed for a judge looking at a laptop screen at 1280–1440 px wide.

## Palette (hex)

| Token | Value | Use |
|---|---|---|
| `canvas` | `#F6F7F5` | Warm off-white overall background |
| `surface` | `#FFFFFF` | Code and result surfaces |
| `surface-muted` | `#F0F2F1` | Soft header, labels, selected areas |
| `ink` | `#171C22` | Main text |
| `ink-muted` | `#68727B` | Descriptive text |
| `line` | `#DDE2E3` | Dividers, thin borders |
| `accent` | `#2E5BEF` | Primary CTA, focused controls |
| `accent-soft` | `#EBF0FF` | Subtle blue selection |
| `positive` | `#217451` | Same behaviour (not proof of safety) |
| `positive-soft` | `#E8F4EC` | Soft same-behaviour background |
| `danger` | `#B83F48` | Behaviour changed |
| `danger-soft` | `#FBEDEF` | Soft changed-behaviour background |
| `warning` | `#94611A` | Execution/model warning |
| `warning-soft` | `#FFF3DF` | Soft warning background |

Use colour **with text labels**; do not depend on colour alone. The brand accent is blue, **not neon purple**. No gradients or glassmorphism.

## Typography

- Use macOS system sans stack: `-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', sans-serif` where custom CSS can support it; no external fonts required.
- Use monospace stack: `'SFMono-Regular', Menlo, Consolas, monospace` for code and numeric evidence.
- Main heading: 30–34 px, 650–700 weight, line-height 1.15. No huge 72 px hero.
- Section titles: 17–20 px semibold.
- Body: 14–15 px, 1.45–1.6 line-height.
- Labels: 11–12 px, medium/semibold, slight letterspacing where appropriate.
- Code: 12.5–13.5 px with accessible contrast and enough line height.
- Tabular numeric metrics in monospace or tabular-numeral feature.

## Spacing and geometry

- 8 px spacing system (4, 8, 12, 16, 24, 32), not random gaps.
- App maximum content width around 1320 px, with 24–40 px page gutters.
- Surface radius 8–12 px; buttons 8 px; avoid pill-shaped everything.
- Thin 1 px borders. Use shadows only for genuine elevation (e.g., floating modal), not every card.
- Two code panes of equal height and width; narrow gutter 16–20 px.
- Results table rows 44–52 px for scanning, with tidy alignment.

## Compositional layout

### Header (quiet, 64–72 px)

Left: small custom text brand mark `R/` plus `RippleTest` and thin separator / `Differential testing`. Right: lightweight model-status indicator `LOCAL / QWEN2.5-CODER 7B` with readable ready/error status.

### Intro/controls (compact)

Small eyebrow `WORKSPACE 01 / REGRESSION ANALYSIS`; main title `Find the break before it ships.`; one-sentence explainer. Near the title or immediately below: scenario selector and **one** primary CTA `Generate & run tests`.

### Contract strip

One horizontal, lightweight text row: `INTENDED BEHAVIOUR` and contract statement. Provides a clear oracle/context but does not automatically declare all differences bugs.

### Twin code panes

Labels: `01 / ORIGINAL` and `02 / MODIFIED`. Technical caption with function name. Read-only Python with line numbers using native Streamlit code when possible. For changed lines use small, accessible diff annotations as enhancement, not brittle syntax-highlight rewriting.

### Results summary

A slim row or grid of four restrained metrics: `Executed`, `Same`, `Changed`, `Errors`. **Don't build four oversized bento cards**. A tiny coloured marker is sufficient for changed/error counts.

### Outcome matrix

This is the main showcase. Table columns: `#`, `Input`, `Original`, `Modified`, `Outcome`. Changed rows have subtle soft-rose background or thin left border. Clicking/expanding a row reveals *why the AI proposed it* and exact evidence. Sort changed first if UX benefits, but preserve stable test-case numbering.

### Provenance and limitations

Footer / caption: `Test inputs: Qwen2.5-Coder 7B • Local Ollama` and an unobtrusive but legible warning `Changed behaviour may be intentional. No difference found ≠ proof of correctness.`

## Interactive states

- Ready: model status dot + text, no confetti or celebratory graphics.
- Generating: disable duplicate trigger where possible; show honest stage text.
- Changed: semantic `Needs review`/`Changed` chips; don't imply “security critical”.
- Same: neutral green `Same result`.
- Error: concise diagnostic and retry.
- Empty: a useful calm prompt, not a huge illustration or fake graphs.

## UX accessibility / refinement

- AA-like contrast goals for text where possible, strong visible focus, 44 px clickable target for primary CTA.
- Keyboard reachable controls and meaningful component labels.
- Streamlit reruns shouldn't wipe a selected test or visually jump excessively.
- At <900 px, code panels stack vertically and summaries wrap without sideways scrolling of controls.
- Responsive typography: keep title under ~34 px desktop, ~27 px narrow.
- Deliberate code truncation? **No**: show full trusted 2–4 line examples and allow scrolling for future scenarios.
- A clear one-screen narrative: select -> compare -> generate -> inspect. Avoid tabs and modal-heavy flows.

## Streamlit implementation guidance

- Use `st.set_page_config(layout="wide")`, `.streamlit/config.toml` official theme options, `st.columns`, `st.container`, `st.code`, `st.dataframe` or appropriately accessible layout to get reliable interactions.
- Apply small targeted CSS for stable wrappers; **do not rely on brittle, undocumented internal class selectors** to simulate a frontend framework.
- Escape/encode all dynamic model output in any raw HTML. Safer default: render outputs using native Streamlit components.
- Skip external images, animation libraries, and font downloads. The core feature remains usable even if CSS tweaks fail.

## What to avoid, explicitly

- Purple/pink/orange gradient heroes, cyberpunk grids, aurora blobs, glossy glass cards, excessive blur and rounded chips.
- Generic robot/head silhouettes, sparkles everywhere, 3D icons, abstract chart art, and huge illustrations.
- Fake “98% confidence”, “AI analysed 28 files” when no such work happened, invented repository data, demo masquerading as user repo.
- Crowded sidebars with irrelevant features or empty placeholders for future functionality.
- Shiny yet unusable UI that sacrifices readable code or actionable evidence.

## Design acceptance test

At 1440×900, without scrolling too far, a first-time viewer can identify product name, selected scenario, both function versions, model status, the action, and—after running—whether any changed behaviour was observed. The screen should be screenshot-worthy because of restraint, contrast, alignment and useful information, not because of decorative effects.
