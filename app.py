"""RippleTest 2.0 — AI-assisted code-change investigation.

A developer pastes two versions of a small numeric Python function. A local
open-weight model proposes test inputs; the application executes both versions
with identical, validated arguments and reports exactly what changed.

Safety: the edited text is either recognised as a bundled, reviewed example
(executed by import) or executed by a restricted AST interpreter. There is no
exec, no eval, no import, and no function call into pasted code, ever.
"""

from __future__ import annotations

import difflib
import json
from datetime import datetime
from html import escape
from typing import Any

import streamlit as st

from core import ollama_client, pipeline
from core.differential import CaseResult
from core.expression import (
    ENGINE_INTERPRETER,
    ENGINE_LABELS,
    ENGINE_NOTES,
    ENGINE_TRUSTED,
    EditError,
    engine_for,
    resolve_execution_target,
)
from core.ollama_client import MODEL_NAME, ModelStatus
from core.report import run_report
from core.scenarios import SCENARIOS, Scenario, get_scenario, scenario_matching_sources
from core.validators import (
    ModelResponseError,
    RejectedCase,
    parse_json_content,
    validate_cases,
)

st.set_page_config(
    page_title="RippleTest — Code-change investigation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design system: Midnight Precision
# ---------------------------------------------------------------------------

CSS = """
:root {
  --canvas: #0B0F17;
  --surface: #151D2B;
  --surface-deep: #101724;
  --surface-raised: #1B2537;
  --line: #23304A;
  --line-soft: #1C2740;
  --ink: #E8ECF4;
  --ink-muted: #8B98AD;
  --ink-faint: #64748B;
  --accent: #4169FF;
  --accent-hover: #5B7CFF;
  --accent-soft: rgba(65, 105, 255, 0.16);
  --accent-line: rgba(65, 105, 255, 0.42);
  --positive: #4ADE9B;
  --positive-soft: rgba(74, 222, 155, 0.13);
  --positive-line: rgba(74, 222, 155, 0.34);
  --danger: #FF6B6B;
  --danger-soft: rgba(255, 107, 107, 0.12);
  --danger-line: rgba(255, 107, 107, 0.36);
  --warning: #E0A458;
  --warning-soft: rgba(224, 164, 88, 0.12);
  --mono: "SFMono-Regular", "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  --sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--canvas);
  color: var(--ink);
  font-family: var(--sans);
}
[data-testid="stAppViewContainer"] { font-size: 15px; }
[data-testid="stMainBlockContainer"] {
  max-width: 1340px;
  padding: 1.25rem 40px 5rem;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

h1, h2, h3, h4, h5, h6 { color: var(--ink); }
code, pre, kbd { font-family: var(--mono); }

/* Typography surfaces ------------------------------------------------- */
.rt-header, .rt-header *, .rt-eyebrow, .rt-title, .rt-lede,
.rt-contract, .rt-contract *, .rt-pane-head, .rt-pane-label, .rt-pane-fn,
.rt-section, .rt-section *, .rt-summary, .rt-summary *,
.rt-table-wrap, .rt-table-wrap *, .rt-inspector, .rt-inspector *,
.rt-notice, .rt-notice *, .rt-empty, .rt-empty *, .rt-footer, .rt-footer *,
.rt-detail-grid, .rt-detail-grid *, .rt-rejects, .rt-rejects *,
.rt-control-note, .rt-prov, .rt-chip, .rt-highlight, .rt-highlight *,
.rt-sig, .rt-sig *, .rt-toolbar, .rt-toolbar * {
  font-family: var(--sans);
}
.rt-eyebrow, .rt-pane-label, .rt-metric-label, .rt-field-label,
.rt-notice-label, .rt-table th, .rt-eyebrow, .rt-prov, .rt-chip,
.rt-detail-label, .rt-footer .rt-mono, .rt-sig, .rt-pane-fn,
.rt-metric-value, .rt-table .num, .rt-table .mono, .rt-obs-value,
.rt-obs-sub, .rt-args, .rt-delta, .rt-detail-value, .rt-rejects code,
.rt-rejects li b, .rt-status, .rt-mark {
  font-family: var(--mono);
}

/* Header --------------------------------------------------------------- */
.rt-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 20px; padding: 6px 0 18px; margin-bottom: 30px;
  border-bottom: 1px solid var(--line-soft);
}
.rt-brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
.rt-mark {
  font-size: 13px; font-weight: 600; letter-spacing: -.02em;
  color: var(--canvas); background: var(--ink); border-radius: 5px;
  padding: 3px 7px 4px; line-height: 1;
}
.rt-brand-name { font-size: 15.5px; font-weight: 640; letter-spacing: -.012em; }
.rt-brand-sub {
  font-size: 9.5px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
  color: var(--ink-faint); border-left: 1px solid var(--line); padding-left: 12px;
}
.rt-status {
  display: inline-flex; align-items: center; gap: 8px; white-space: nowrap;
  font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--ink-muted); cursor: default;
}
.rt-dot { width: 6px; height: 6px; border-radius: 50%; flex: none; }
.rt-dot--ok { background: var(--positive); box-shadow: 0 0 0 3px var(--positive-soft); }
.rt-dot--bad { background: var(--danger); box-shadow: 0 0 0 3px var(--danger-soft); }
.rt-dot--wait { background: var(--warning); box-shadow: 0 0 0 3px var(--warning-soft); }

/* Intro ---------------------------------------------------------------- */
.rt-eyebrow {
  font-size: 10px; letter-spacing: .16em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 12px; font-weight: 600;
}
.rt-eyebrow .sep { color: var(--ink-faint); padding: 0 6px; }
.rt-title {
  font-family: var(--sans) !important;
  font-size: 31px !important; font-weight: 680 !important; letter-spacing: -.028em !important;
  line-height: 1.14 !important; margin: 0 0 10px !important; color: var(--ink) !important;
}
h2.rt-section-title {
  font-family: var(--sans) !important; font-size: 17px !important;
  font-weight: 650 !important; letter-spacing: -.012em !important;
  margin: 0 !important; color: var(--ink) !important;
}
.rt-lede {
  font-size: 14.5px !important; line-height: 1.55 !important;
  color: var(--ink-muted) !important; max-width: 60ch; margin: 0 !important;
}
.rt-lede code {
  font-size: 12px; background: var(--surface); color: var(--accent-hover);
  padding: 1px 5px; border-radius: 4px; border: 1px solid var(--line-soft);
}

/* Controls ------------------------------------------------------------- */
.rt-toolbar { display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; }
.rt-control-note { font-size: 11px; line-height: 1.45; color: var(--ink-faint); margin-top: 6px; }
.rt-control-note b { color: var(--ink-muted); font-weight: 600; }

[data-testid="stWidgetLabel"] p {
  font-family: var(--sans) !important;
  font-size: 10px !important; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ink-faint) !important; font-weight: 600 !important;
}
[data-baseweb="select"] > div {
  background: var(--surface) !important; border: 1px solid var(--line) !important;
  border-radius: 8px !important; min-height: 40px !important;
  font-size: 13px !important; color: var(--ink) !important;
}
[data-baseweb="select"] > div:hover { border-color: var(--accent-line) !important; }
[data-baseweb="select"] svg { color: var(--ink-muted); }

[data-testid="stBaseButton-primary"], button[kind="primary"] {
  background: var(--accent) !important; border: 1px solid var(--accent) !important;
  color: #FFFFFF !important; border-radius: 8px !important;
  font-size: 13.5px !important; font-weight: 600 !important;
  min-height: 44px !important; box-shadow: var(--shadow) !important;
  transition: background .14s ease, transform .1s ease;
}
[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover {
  background: var(--accent-hover) !important; border-color: var(--accent-hover) !important;
}
[data-testid="stBaseButton-primary"]:active { transform: translateY(1px); }
[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
  background: transparent !important; border: 1px solid var(--line) !important;
  color: var(--ink-muted) !important; border-radius: 8px !important;
  font-size: 12.5px !important; font-weight: 550 !important;
  min-height: 40px !important; box-shadow: none !important;
  transition: border-color .14s ease, color .14s ease;
}
[data-testid="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover {
  border-color: var(--accent-line) !important; color: var(--ink) !important;
}
button:focus-visible, [data-baseweb="select"]:focus-within,
[data-testid="stTextArea"] textarea:focus {
  outline: 2px solid var(--accent) !important; outline-offset: 2px !important;
}
[data-testid="stExpander"] {
  border: 1px solid var(--line-soft) !important; border-radius: 10px !important;
  background: var(--surface) !important; box-shadow: none !important;
}
[data-testid="stExpander"] summary {
  font-size: 12.5px !important; font-weight: 550 !important; color: var(--ink-muted) !important;
}
[data-testid="stExpander"] summary:hover { color: var(--ink) !important; }
[data-testid="stCaptionContainer"] p, .stCaption p {
  font-size: 11px !important; color: var(--ink-faint) !important; line-height: 1.5;
}

/* Code workspace ------------------------------------------------------- */
.rt-panes { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.rt-pane {
  background: var(--surface-deep); border: 1px solid var(--line);
  border-radius: 12px; overflow: hidden; display: flex; flex-direction: column;
  min-width: 0; transition: border-color .16s ease;
}
.rt-pane:focus-within { border-color: var(--accent-line); }
.rt-pane-head {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 10px 14px; border-bottom: 1px solid var(--line-soft);
  background: var(--surface); min-height: 40px;
}
.rt-pane-label {
  font-size: 10px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ink-muted); white-space: nowrap;
}
.rt-pane-label b { color: var(--ink); font-weight: 600; }
.rt-pane-fn {
  font-size: 10px; color: var(--ink-faint); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}
.rt-editor { padding: 0; }
[data-testid="stTextArea"] { background: transparent; }
[data-testid="stTextArea"] textarea {
  background: transparent !important; color: var(--ink) !important;
  font-family: var(--mono) !important; font-size: 12.75px !important;
  line-height: 21px !important; border: none !important;
  padding: 14px 16px 16px !important; caret-color: var(--accent) !important;
  resize: vertical; min-height: 190px;
}
[data-testid="stTextArea"] textarea::placeholder { color: var(--ink-faint); }
.rt-pane-foot {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 8px 14px; border-top: 1px solid var(--line-soft); background: var(--surface);
  font-family: var(--mono); font-size: 10px; letter-spacing: .04em; color: var(--ink-faint);
}
.rt-sig { font-family: var(--mono); font-size: 10px; color: var(--ink-muted); }
.rt-sig .ok { color: var(--positive); }
.rt-sig .bad { color: var(--danger); }
.rt-diffcount { color: var(--ink-faint); }
.rt-diffcount .n { color: var(--accent-hover); }

/* Sections ------------------------------------------------------------- */
.rt-section {
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 18px; margin: 32px 0 14px;
}
.rt-section-sub { font-size: 12px; color: var(--ink-muted); margin-top: 4px; }
.rt-prov-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.rt-prov {
  font-size: 9px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--ink-muted); background: var(--surface);
  border: 1px solid var(--line); border-radius: 5px; padding: 3px 8px; white-space: nowrap;
}
.rt-prov--ai { color: var(--accent-hover); background: var(--accent-soft); border-color: var(--accent-line); }
.rt-prov--engine { color: var(--positive); background: var(--positive-soft); border-color: var(--positive-line); }

/* Summary -------------------------------------------------------------- */
.rt-summary {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  overflow: hidden; margin-bottom: 16px;
}
.rt-metric { padding: 12px 18px 13px; border-left: 1px solid var(--line-soft); }
.rt-metric:first-child { border-left: 0; }
.rt-metric-label {
  display: flex; align-items: center; gap: 6px;
  font-size: 9.5px; letter-spacing: .11em; text-transform: uppercase;
  color: var(--ink-faint); font-weight: 600; white-space: nowrap;
}
.rt-metric-value {
  font-family: var(--mono); font-size: 22px; font-weight: 500;
  font-variant-numeric: tabular-nums; line-height: 1.3; margin-top: 2px; color: var(--ink);
}
.rt-metric--changed .rt-metric-value { color: var(--danger); }
.rt-metric--error .rt-metric-value { color: var(--warning); }
.rt-flag { width: 5px; height: 5px; border-radius: 50%; flex: none; }
.rt-flag--danger { background: var(--danger); }
.rt-flag--warning { background: var(--warning); }

/* Strongest counterexample --------------------------------------------- */
.rt-highlight {
  border: 1px solid var(--accent-line); border-radius: 12px;
  background: linear-gradient(0deg, var(--accent-soft), var(--accent-soft)), var(--surface);
  padding: 18px 20px; margin-bottom: 16px; box-shadow: var(--shadow);
}
.rt-highlight--none {
  border-color: var(--line); background: var(--surface); box-shadow: none;
}
.rt-highlight-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 14px; margin-bottom: 14px; flex-wrap: wrap;
}
.rt-highlight-eyebrow {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--accent-hover); font-weight: 600;
}
.rt-highlight--none .rt-highlight-eyebrow { color: var(--ink-faint); }
.rt-compare {
  display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 16px; align-items: center;
}
.rt-side { min-width: 0; }
.rt-side-label {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .12em;
  text-transform: uppercase; color: var(--ink-faint); margin-bottom: 6px;
}
.rt-side-value {
  font-family: var(--mono); font-size: 26px; font-variant-numeric: tabular-nums;
  color: var(--ink); line-height: 1.15; word-break: break-all;
}
.rt-side--old .rt-side-value { color: var(--ink); }
.rt-side--new .rt-side-value { color: var(--danger); }
.rt-arrow { color: var(--ink-faint); font-family: var(--mono); font-size: 18px; }
.rt-highlight-input {
  font-family: var(--mono); font-size: 12px; color: var(--ink-muted);
  margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line-soft);
  overflow-x: auto; white-space: nowrap;
}
.rt-highlight-input b { color: var(--ink); font-weight: 500; }
.rt-highlight-input .delta { color: var(--danger); }
.rt-highlight-reason { font-size: 12px; color: var(--ink-muted); margin-top: 6px; }

/* Table ---------------------------------------------------------------- */
.rt-table-wrap {
  border: 1px solid var(--line); border-radius: 12px;
  background: var(--surface); overflow: hidden;
}
.rt-table-scroll { overflow-x: auto; }
table.rt-table { width: 100%; border-collapse: collapse; min-width: 640px; }
.rt-table thead th {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ink-faint); text-align: left; padding: 10px 16px;
  background: var(--surface-raised); border-bottom: 1px solid var(--line);
  font-weight: 600; white-space: nowrap;
}
.rt-table td {
  padding: 0 16px; height: 46px; border-top: 1px solid var(--line-soft);
  font-size: 12.5px; vertical-align: middle;
}
.rt-table tbody tr:first-child td { border-top: 0; }
.rt-table tbody tr { transition: background .12s ease; }
.rt-table tbody tr:hover td { background: var(--surface-raised); }
.rt-table .num { width: 46px; font-family: var(--mono); font-size: 11px; color: var(--ink-faint); }
.rt-table .mono {
  font-family: var(--mono); font-size: 12.25px; font-variant-numeric: tabular-nums;
  color: var(--ink); white-space: nowrap;
}
.rt-table .out { width: 148px; }
.rt-table .empty { color: var(--ink-faint); }
.rt-row--changed td { background: var(--danger-soft); }
.rt-row--changed:hover td { background: rgba(255, 107, 107, 0.18) !important; }
.rt-row--changed td:first-child { box-shadow: inset 2px 0 0 var(--danger); }
.rt-row--selected td { background: var(--accent-soft) !important; }
.rt-row--selected td:first-child { box-shadow: inset 2px 0 0 var(--accent); }
.rt-exc { color: var(--warning); font-family: var(--mono); font-size: 12px; }

.rt-chip {
  display: inline-flex; align-items: center; gap: 6px; border-radius: 5px;
  font-size: 10.5px; font-weight: 600; padding: 3px 8px; white-space: nowrap;
  border: 1px solid transparent; font-family: var(--sans);
}
.rt-chip .rt-dot { width: 5px; height: 5px; box-shadow: none; }
.rt-chip--same { color: var(--positive); background: var(--positive-soft); border-color: var(--positive-line); }
.rt-chip--same .rt-dot { background: var(--positive); }
.rt-chip--changed { color: var(--danger); background: var(--danger-soft); border-color: var(--danger-line); }
.rt-chip--changed .rt-dot { background: var(--danger); }
.rt-chip--error { color: var(--warning); background: var(--warning-soft); border-color: rgba(224,164,88,.34); }
.rt-chip--error .rt-dot { background: var(--warning); }
.rt-table .rt-chip--changed { background: transparent; }

/* Inspector ------------------------------------------------------------ */
.rt-inspector {
  border: 1px solid var(--line); border-radius: 12px; background: var(--surface);
  padding: 18px 20px 19px; margin-top: 12px;
}
.rt-inspector-head {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 16px; border-bottom: 1px solid var(--line-soft);
  padding-bottom: 13px; margin-bottom: 16px;
}
.rt-inspector-title { font-size: 14.5px; font-weight: 650; letter-spacing: -.008em; }
.rt-inspector-title span { color: var(--ink-faint); font-weight: 500; }
.rt-field-label {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ink-faint); font-weight: 600; margin-bottom: 7px;
}
.rt-args {
  font-family: var(--mono); font-size: 12.8px; color: var(--ink);
  background: var(--surface-deep); border: 1px solid var(--line-soft);
  border-radius: 7px; padding: 10px 12px; margin-bottom: 14px;
  overflow-x: auto; white-space: nowrap;
}
.rt-args b { font-weight: 400; color: var(--ink-muted); }
.rt-grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.rt-obs {
  border: 1px solid var(--line-soft); border-radius: 8px; padding: 11px 13px 12px;
  background: var(--surface-deep);
}
.rt-obs-value {
  font-family: var(--mono); font-size: 15.5px; font-variant-numeric: tabular-nums;
  color: var(--ink); word-break: break-word;
}
.rt-obs--error { border-color: rgba(224,164,88,.3); }
.rt-obs--error .rt-obs-value { color: var(--warning); font-size: 13.5px; }
.rt-obs-sub { font-family: var(--mono); font-size: 10.5px; color: var(--ink-faint); margin-top: 4px; }
.rt-delta { font-family: var(--mono); font-size: 11.5px; color: var(--ink-muted); margin-top: 11px; }
.rt-delta b { color: var(--ink); font-weight: 500; }
.rt-rationale { margin-top: 15px; padding-top: 13px; border-top: 1px solid var(--line-soft); }
.rt-rationale p { margin: 0; font-size: 13px; line-height: 1.55; color: var(--ink); }
.rt-subnote { font-size: 11px; color: var(--ink-faint); margin-top: 6px; display: block; }
.rt-note { font-size: 12px; color: var(--warning); margin-top: 10px; }

/* Notices, empty, details ---------------------------------------------- */
.rt-notice {
  border: 1px solid var(--line); border-left: 3px solid var(--accent);
  background: var(--surface); border-radius: 10px; padding: 14px 17px; margin: 0 0 16px;
}
.rt-notice--error, .rt-notice--changed { border-left-color: var(--danger); }
.rt-notice--warning { border-left-color: var(--warning); }
.rt-notice-label {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ink-faint); font-weight: 600; margin-bottom: 6px;
}
.rt-notice p { margin: 0; font-size: 13px; line-height: 1.55; color: var(--ink); }
.rt-notice .rt-notice-extra {
  margin-top: 7px; font-family: var(--mono); font-size: 10.5px;
  color: var(--ink-muted); line-height: 1.5;
}
.rt-notice code {
  font-size: 11px; background: var(--surface-deep); color: var(--accent-hover);
  padding: 1px 5px; border-radius: 4px; border: 1px solid var(--line-soft);
}
.rt-empty {
  border: 1px dashed var(--line); border-radius: 12px; background: var(--surface);
  padding: 28px 24px; margin-top: 10px;
}
.rt-empty-title { font-size: 14.5px; font-weight: 600; margin-bottom: 7px; color: var(--ink); }
.rt-empty p {
  margin: 0; font-size: 13px; line-height: 1.6; color: var(--ink-muted); max-width: 80ch;
}

.rt-detail-grid {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 15px 20px; margin-bottom: 4px;
}
.rt-detail-label {
  font-family: var(--mono); font-size: 9px; letter-spacing: .11em; text-transform: uppercase;
  color: var(--ink-faint); font-weight: 600; margin-bottom: 4px;
}
.rt-detail-value { font-family: var(--mono); font-size: 12px; color: var(--ink); }
.rt-rejects { margin: 12px 0 0; padding: 0; list-style: none; }
.rt-rejects li {
  border-top: 1px solid var(--line-soft); padding: 9px 0;
  font-size: 12px; color: var(--ink-muted); line-height: 1.5;
}
.rt-rejects li b { font-family: var(--mono); font-weight: 500; color: var(--ink-faint); }
.rt-rejects code {
  display: block; font-family: var(--mono); font-size: 10.5px;
  color: var(--ink-faint); margin-top: 3px; overflow-wrap: anywhere;
}

/* Footer --------------------------------------------------------------- */
.rt-footer {
  display: flex; justify-content: space-between; align-items: flex-start; gap: 24px;
  margin-top: 38px; padding-top: 16px; border-top: 1px solid var(--line-soft);
  font-size: 11px; line-height: 1.55; color: var(--ink-faint);
}
.rt-footer b { color: var(--ink-muted); font-weight: 600; }
.rt-footer .rt-foot-right { text-align: right; }
.rt-footer .rt-mono { font-family: var(--mono); font-size: 10px; }

/* Responsive ----------------------------------------------------------- */
@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"] { padding: 1rem 18px 3.5rem; }
  .rt-header { flex-direction: column; align-items: flex-start; gap: 10px; margin-bottom: 22px; }
  .rt-brand-sub { display: none; }
  .rt-title { font-size: 26px !important; }
  .rt-panes, .rt-grid-2 { grid-template-columns: minmax(0, 1fr); }
  .rt-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .rt-metric:nth-child(3) { border-left: 0; border-top: 1px solid var(--line-soft); }
  .rt-metric:nth-child(4) { border-top: 1px solid var(--line-soft); }
  .rt-section { flex-direction: column; align-items: flex-start; gap: 8px; }
  .rt-prov-row { justify-content: flex-start; }
  .rt-compare { grid-template-columns: minmax(0, 1fr); gap: 10px; }
  .rt-arrow { transform: rotate(90deg); justify-self: start; }
  .rt-footer { flex-direction: column; gap: 10px; }
  .rt-footer .rt-foot-right { text-align: left; }
  .rt-detail-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .rt-side-value { font-size: 21px; }
}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def render(fragment: str) -> None:
    st.markdown(fragment, unsafe_allow_html=True)


def esc(value: object) -> str:
    return escape(str(value), quote=True)


def inline_code(text: str) -> str:
    import re

    return re.sub(r"`([^`]+)`", r"<code>\1</code>", esc(text))


def format_argument(value: Any) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


@st.cache_data(ttl=8, show_spinner=False)
def read_model_status() -> ModelStatus:
    """Cached real health check; short TTL so the header reflects reality."""
    return ollama_client.check_model()


def header_markup(status: ModelStatus | None = None) -> str:
    if status is None:
        dot, text = "rt-dot--wait", "Local / Qwen2.5-Coder 7B · Checking"
        detail = "Checking Ollama at localhost:11434."
    elif status.state == "available":
        dot, text, detail = "rt-dot--ok", "Local / Qwen2.5-Coder 7B · Ready", status.detail
    elif status.state == "model_missing":
        dot, text, detail = "rt-dot--bad", "Ollama / model missing", status.detail
    else:
        dot, text, detail = "rt-dot--bad", "Ollama / offline", status.detail
    return (
        f'<div class="rt-header">'
        f'<div class="rt-brand"><span class="rt-mark">R/</span>'
        f'<span class="rt-brand-name">RippleTest</span>'
        f'<span class="rt-brand-sub">Code-change intelligence</span></div>'
        f'<span class="rt-status" title="{esc(detail)}">'
        f'<span class="rt-dot {dot}"></span>{esc(text)}</span>'
        f"</div>"
    )


def notice_markup(label: str, message: str, *, variant: str = "info", extra: str = "") -> str:
    tone = {
        "info": "",
        "error": " rt-notice--error",
        "changed": " rt-notice--changed",
        "warning": " rt-notice--warning",
    }[variant]
    extra_html = f'<div class="rt-notice-extra">{esc(extra)}</div>' if extra else ""
    return (
        f'<div class="rt-notice{tone}">'
        f'<div class="rt-notice-label">{esc(label)}</div>'
        f"<p>{message}</p>{extra_html}</div>"
    )


def status_notice(status: ModelStatus) -> str:
    if status.state == "model_missing":
        return notice_markup(
            "Local model not installed",
            f"Ollama is running, but <code>{esc(MODEL_NAME)}</code> is not installed. "
            f"Run <code>ollama pull {esc(MODEL_NAME)}</code>, then check again.",
            variant="error",
            extra=status.detail,
        )
    return notice_markup(
        "Ollama unavailable",
        "Ollama is not reachable at <code>http://localhost:11434</code>. Start Ollama and "
        "check the connection again. No hosted model will be substituted.",
        variant="error",
        extra=status.detail,
    )


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

CHIPS = {
    "same": ("rt-chip--same", "Same result"),
    "changed": ("rt-chip--changed", "Behaviour changed"),
    "error": ("rt-chip--error", "Execution error"),
}
OUTCOME_ORDER = {"changed": 0, "error": 1, "same": 2}


def chip(classification: str) -> str:
    tone, label = CHIPS[classification]
    return f'<span class="rt-chip {tone}"><span class="rt-dot"></span>{label}</span>'


def outcome_cell(outcome) -> str:
    if outcome.kind == "exception":
        return f'<span class="rt-exc">{esc(outcome.exception_type)}</span>'
    return esc(outcome.display)


def sort_results(results: tuple[CaseResult, ...]) -> list[CaseResult]:
    return sorted(results, key=lambda item: (OUTCOME_ORDER[item.classification], item.number))


def summary_markup(summary) -> str:
    metrics = [
        ("Executed", summary.executed, ""),
        ("Same", summary.same, ""),
        ("Changed", summary.changed, "changed"),
        ("Errors", summary.errors, "error"),
    ]
    cells = []
    for label, value, tone in metrics:
        flag = ""
        if tone == "changed" and value:
            flag = '<span class="rt-flag rt-flag--danger"></span>'
        if tone == "error" and value:
            flag = '<span class="rt-flag rt-flag--warning"></span>'
        modifier = f" rt-metric--{tone}" if tone and value else ""
        cells.append(
            f'<div class="rt-metric{modifier}">'
            f'<div class="rt-metric-label">{flag}{label}</div>'
            f'<div class="rt-metric-value">{value}</div>'
            f"</div>"
        )
    return f'<div class="rt-summary">{"".join(cells)}</div>'


def strongest_counterexample(results: tuple[CaseResult, ...]) -> CaseResult | None:
    """The changed case with the largest numeric swing. Real data, no scoring."""
    changed = [item for item in results if item.classification == "changed"]
    if not changed:
        return None

    def magnitude(item: CaseResult) -> float:
        if item.delta is not None:
            return abs(item.delta)
        # Non-numeric divergence: fall back to the observed string difference.
        return float(len(item.modified.display) + len(item.original.display))

    return max(changed, key=magnitude)


def highlight_markup(results: tuple[CaseResult, ...], function_name: str) -> str:
    best = strongest_counterexample(results)
    if best is None:
        summary_errors = sum(1 for item in results if item.classification == "error")
        if summary_errors:
            body = (
                f"{summary_errors} case(s) raised an exception. No completed divergence was "
                "observed, so there is no counterexample to show."
            )
        else:
            body = (
                f"All {len(results)} executed cases returned matching values. "
                "This does not establish that the change is regression-free."
            )
        return (
            '<div class="rt-highlight rt-highlight--none">'
            '<div class="rt-highlight-head">'
            '<span class="rt-highlight-eyebrow">Strongest observed divergence</span>'
            "</div>"
            f"<p style='margin:0;font-size:13px;line-height:1.55;color:var(--ink-muted)'>{body}</p>"
            "</div>"
        )

    arguments = ", ".join(
        f"{name}={format_argument(value)}" for name, value in best.arguments.items()
    )
    delta = f"Δ {best.delta:+.12g}" if best.delta is not None else "values differ"
    old_value = (
        best.original.display if best.original.kind == "value" else best.original.exception_type
    )
    new_value = (
        best.modified.display if best.modified.kind == "value" else best.modified.exception_type
    )
    return (
        '<div class="rt-highlight">'
        '<div class="rt-highlight-head">'
        '<span class="rt-highlight-eyebrow">Strongest observed divergence</span>'
        f"{chip(best.classification)}</div>"
        '<div class="rt-compare">'
        f'<div class="rt-side rt-side--old"><div class="rt-side-label">01 / Original</div>'
        f'<div class="rt-side-value">{esc(old_value)}</div></div>'
        '<div class="rt-arrow">→</div>'
        f'<div class="rt-side rt-side--new"><div class="rt-side-label">02 / Modified</div>'
        f'<div class="rt-side-value">{esc(new_value)}</div></div>'
        "</div>"
        f'<div class="rt-highlight-input">case #{best.number:02d} · <b>{esc(arguments)}</b>'
        f' &nbsp;·&nbsp; <span class="delta">{esc(delta)}</span>'
        f" &nbsp;·&nbsp; {esc(function_name)}</div>"
        f'<div class="rt-highlight-reason">{esc(best.reason)}</div>'
        "</div>"
    )


def table_markup(results: tuple[CaseResult, ...], selected_number: int) -> str:
    rows = []
    for result in sort_results(results):
        classes = ["rt-row"]
        if result.classification == "changed":
            classes.append("rt-row--changed")
        if result.number == selected_number:
            classes.append("rt-row--selected")
        arguments = ", ".join(
            f"{name}={format_argument(value)}" for name, value in result.arguments.items()
        )
        rows.append(
            f'<tr class="{" ".join(classes)}">'
            f'<td class="num">{result.number:02d}</td>'
            f'<td class="mono">{esc(arguments)}</td>'
            f'<td class="mono">{outcome_cell(result.original)}</td>'
            f'<td class="mono">{outcome_cell(result.modified)}</td>'
            f'<td class="out">{chip(result.classification)}</td>'
            f"</tr>"
        )
    return (
        '<div class="rt-table-wrap"><div class="rt-table-scroll">'
        '<table class="rt-table"><thead><tr>'
        '<th scope="col">#</th><th scope="col">Input</th>'
        '<th scope="col">Original</th><th scope="col">Modified</th>'
        '<th scope="col">Outcome</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div></div>"
    )


def inspector_markup(function_name: str, result: CaseResult) -> str:
    arguments = " &nbsp; ".join(
        f"<b>{esc(name)}</b> = {esc(format_argument(value))}"
        for name, value in result.arguments.items()
    )

    def observation(label: str, outcome) -> str:
        if outcome.kind == "exception":
            note = outcome.exception_message or ""
            return (
                f'<div class="rt-obs rt-obs--error">'
                f'<div class="rt-field-label">{esc(label)}</div>'
                f'<div class="rt-obs-value">{esc(outcome.exception_type)}</div>'
                f'<div class="rt-obs-sub">{esc(note)}</div></div>'
            )
        return (
            f'<div class="rt-obs"><div class="rt-field-label">{esc(label)}</div>'
            f'<div class="rt-obs-value">{esc(outcome.display)}</div>'
            f'<div class="rt-obs-sub">{esc(type(outcome.value).__name__)}</div></div>'
        )

    delta = f'<div class="rt-delta">modified − original = <b>{result.delta:+.12g}</b></div>' if result.delta is not None else ""
    if result.provenance == "ai":
        reason_label = "Why the model proposed it"
        subnote = "Model suggestion, not execution evidence. The outputs above come from Python."
    else:
        reason_label = "Why this baseline case exists"
        subnote = "Pre-written case. This run did not use the model."
    note = f'<div class="rt-note">{esc(result.note)}</div>' if result.note else ""

    return (
        '<div class="rt-inspector">'
        '<div class="rt-inspector-head">'
        f'<div class="rt-inspector-title">Case {result.number:02d} '
        f"<span>/ {esc(function_name)}</span></div>"
        f"{chip(result.classification)}</div>"
        '<div class="rt-field-label">Identical arguments passed to both versions</div>'
        f'<div class="rt-args">{arguments}</div>'
        f'<div class="rt-grid-2">'
        f'{observation("01 / Original result", result.original)}'
        f'{observation("02 / Modified result", result.modified)}'
        f"</div>{delta}{note}"
        f'<div class="rt-rationale"><div class="rt-field-label">{reason_label}</div>'
        f"<p>{esc(result.reason)}</p>"
        f'<span class="rt-subnote">{subnote}</span></div>'
        "</div>"
    )


def rejected_markup(rejected: tuple[RejectedCase, ...]) -> str:
    if not rejected:
        return ""
    items = "".join(
        f"<li><b>Proposal {item.number:02d}</b> — {esc(item.reason)}"
        f"<code>{esc(item.raw)}</code></li>"
        for item in rejected
    )
    return f'<ul class="rt-rejects">{items}</ul>'


def run_details(run) -> None:
    with st.expander("Run details, validation, and export"):
        request_time = f"{run.duration_seconds:.1f}s" if run.duration_seconds is not None else "—"
        engine = run.engine if hasattr(run, "engine") else ENGINE_TRUSTED
        engine_note = ENGINE_NOTES.get(engine, "")
        render(
            '<div class="rt-detail-grid">'
            f'<div><div class="rt-detail-label">Input source</div>'
            f'<div class="rt-detail-value">'
            f'{"AI-generated" if run.provenance == "ai" else "Curated baseline"}</div></div>'
            f'<div><div class="rt-detail-label">Model</div>'
            f'<div class="rt-detail-value">{esc(run.model or "none (curated)")}</div></div>'
            f'<div><div class="rt-detail-label">Request time</div>'
            f'<div class="rt-detail-value">{esc(request_time)}</div></div>'
            f'<div><div class="rt-detail-label">Execution engine</div>'
            f'<div class="rt-detail-value">{esc(run.engine_label)}</div></div>'
            f'<div><div class="rt-detail-label">Comparison tolerance</div>'
            f'<div class="rt-detail-value">rel 1e-9 · abs 1e-9</div></div>'
            f'<div><div class="rt-detail-label">Proposed</div>'
            f'<div class="rt-detail-value">{run.validation.proposed_count}</div></div>'
            f'<div><div class="rt-detail-label">Accepted</div>'
            f'<div class="rt-detail-value">{len(run.validation.accepted)}</div></div>'
            f'<div><div class="rt-detail-label">Rejected</div>'
            f'<div class="rt-detail-value">{len(run.validation.rejected)}</div></div>'
            "</div>"
        )
        st.caption(engine_note)
        if run.validation.rejected:
            render(
                '<div class="rt-field-label" style="margin-top:14px">Rejected proposals</div>'
                + rejected_markup(run.validation.rejected)
            )
        else:
            st.caption("Every proposed case passed validation and was executed.")
        st.download_button(
            "Download JSON report",
            data=json.dumps(run_report(run), indent=2, ensure_ascii=False),
            file_name=f"rippletest-{run.scenario_id}-{run.run_id}.json",
            mime="application/json",
            type="secondary",
        )


def render_run(run) -> None:
    summary = run.summary
    provenance = (
        f'<span class="rt-prov rt-prov--ai">AI-generated inputs</span>'
        f'<span class="rt-prov">{esc(run.model or MODEL_NAME)} · local Ollama</span>'
        if run.provenance == "ai"
        else '<span class="rt-prov">Curated baseline</span>'
        '<span class="rt-prov">pre-written · not AI-generated</span>'
    )
    engine = f'<span class="rt-prov rt-prov--engine">{esc(run.engine_label)}</span>'
    stamp = datetime.fromisoformat(run.created_at).astimezone().strftime("%H:%M")

    render(
        f'<div class="rt-section"><div>'
        f'<h2 class="rt-section-title">Observed behaviour</h2>'
        f'<div class="rt-section-sub">{summary.executed} cases executed at {stamp}. '
        f"Outcomes come from real calls to both versions.</div></div>"
        f'<div class="rt-prov-row">{provenance}{engine}</div></div>'
    )
    render(summary_markup(summary))
    render(highlight_markup(run.results, run.function_name))

    ordered = sort_results(run.results)
    options = [item.number for item in ordered]
    key = f"inspect_{run.run_id}"
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    selected_number = st.session_state[key]

    render(table_markup(run.results, selected_number))
    st.caption(
        "Sorted by outcome; case numbers are stable. Select a case to inspect its evidence."
    )

    head_left, head_right = st.columns([2.2, 1], gap="large")
    with head_left:
        render('<h2 class="rt-section-title" style="margin-top:16px">Case evidence</h2>')
    with head_right:
        st.selectbox(
            "Inspect case",
            options=options,
            key=key,
            format_func=lambda number: next(
                f"Case {number:02d} · {CHIPS[item.classification][1]}"
                for item in run.results
                if item.number == number
            ),
        )

    selected = next(item for item in run.results if item.number == selected_number)
    render(inspector_markup(run.function_name, selected))
    run_details(run)


def render_failure(failure: dict[str, Any]) -> None:
    render(
        notice_markup(
            failure.get("title", "Analysis failed"),
            failure.get("message", "The analysis could not be completed."),
            variant="error",
            extra=failure.get("detail", ""),
        )
    )
    rejected = failure.get("rejected") or ()
    if rejected:
        render(
            '<div class="rt-field-label">Rejected proposals</div>' + rejected_markup(tuple(rejected))
        )
    if failure.get("raw"):
        with st.expander("Raw reply from the local model"):
            st.code(failure["raw"][:4000], language="json")
    if failure.get("kind") in {"offline", "timeout", "model_missing", "http_error", "transport"}:
        if st.button("Check connection", type="secondary"):
            read_model_status.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# Analysis: parse -> model -> validate -> execute -> compare
# ---------------------------------------------------------------------------


def perform_analysis(original_source: str, modified_source: str) -> None:
    st.session_state["failure"] = None
    st.session_state["run"] = None

    with st.status("Analysing change", expanded=True) as box:
        st.write("Parsing both versions")

        try:
            target = resolve_execution_target(original_source, modified_source)
        except EditError as exc:
            box.update(label="Unsupported Python in the editor", state="error")
            st.session_state["failure"] = {
                "kind": "unsupported_syntax",
                "title": "Unsupported Python in the editor",
                "message": esc(
                    "RippleTest only executes a small, safe numeric subset: one function, "
                    "numeric parameters, and a single return expression. The pasted code was "
                    "not executed, and no results were produced."
                ),
                "detail": str(exc),
                "raw": "",
                "rejected": (),
            }
            return

        engine = engine_for(target)
        st.write(f"Requesting {MODEL_NAME} from local Ollama")
        generation = ollama_client.generate_test_cases(target)

        if not generation.ok:
            box.update(label="The local model request failed", state="error")
            st.session_state["failure"] = {
                "kind": generation.error_kind,
                "title": {
                    "offline": "Ollama is offline",
                    "timeout": "The local model timed out",
                    "model_missing": "Model not installed",
                    "empty_response": "AI response invalid",
                    "bad_response": "AI response invalid",
                }.get(generation.error_kind, "The model request failed"),
                "message": esc(generation.error_message),
                "detail": generation.detail,
                "raw": "",
                "rejected": (),
            }
            return

        st.write("Validating proposed inputs")
        try:
            payload = parse_json_content(generation.raw_content)
            validation = validate_cases(target, payload)
        except ModelResponseError as exc:
            box.update(label="The model reply could not be used", state="error")
            st.session_state["failure"] = {
                "kind": exc.kind,
                "title": "AI response invalid",
                "message": esc(exc.message),
                "detail": exc.detail,
                "raw": generation.raw_content,
                "rejected": (),
            }
            return

        if not validation.accepted:
            box.update(label="No usable test inputs", state="error")
            st.session_state["failure"] = {
                "kind": "no_valid_cases",
                "title": "No usable test inputs",
                "message": (
                    f"The model returned {validation.proposed_count} proposals, but none passed "
                    "validation. Nothing was executed. Edit the code or analyse again."
                ),
                "detail": "",
                "raw": generation.raw_content,
                "rejected": validation.rejected,
            }
            return

        st.write(f"Executing with {ENGINE_LABELS.get(engine, engine).lower()}")
        run = pipeline.build_run(
            target,
            validation,
            provenance="ai",
            model=generation.model,
            duration_seconds=generation.duration_seconds,
        )
        st.write("Comparing real outputs")
        box.update(label=f"Compared {len(run.results)} cases", state="complete", expanded=False)

    st.session_state["run"] = run


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

for state_key in ("run", "failure"):
    if state_key not in st.session_state:
        st.session_state[state_key] = None

# Seed the editors with the discount example so the workspace is immediately usable.
if "editor_original" not in st.session_state:
    st.session_state["editor_original"] = SCENARIOS["discount"].original_source
if "editor_modified" not in st.session_state:
    st.session_state["editor_modified"] = SCENARIOS["discount"].modified_source


def load_example() -> None:
    """Populate both editors with a bundled example. Never runs anything."""
    example = SCENARIOS[st.session_state["example_choice"]]
    st.session_state["editor_original"] = example.original_source
    st.session_state["editor_modified"] = example.modified_source
    st.session_state["run"] = None
    st.session_state["failure"] = None


header_slot = st.empty()
header_slot.markdown(header_markup(), unsafe_allow_html=True)
status = read_model_status()
header_slot.markdown(header_markup(status), unsafe_allow_html=True)

intro_col, controls_col = st.columns([1.65, 1], gap="large")
with intro_col:
    render(
        '<div class="rt-eyebrow">Workspace 01<span class="sep">/</span>'
        'Change investigation</div>'
        '<h1 class="rt-title">Find the break before it ships.</h1>'
        '<p class="rt-lede">Paste two versions of a numeric Python function. '
        "Qwen2.5-Coder 7B proposes test inputs on this Mac, and both versions are "
        "executed with the same arguments so you can see exactly what changed.</p>"
    )
with controls_col:
    st.selectbox(
        "Load example",
        options=list(SCENARIOS),
        format_func=lambda key: SCENARIOS[key].label,
        key="example_choice",
        on_change=load_example,
    )
    render(
        '<div class="rt-control-note">Loads a bundled example into the editors. '
        "Nothing runs until you analyse.</div>"
    )

if status.state != "available":
    render(status_notice(status))
    if st.button("Check connection", type="secondary", key="recheck_top"):
        read_model_status.clear()
        st.rerun()
# --- Code workspace --------------------------------------------------------

# Live readout computed from the real text in the panels: signature compatibility
# and how many lines the developer actually changed.
try:
    from core.expression import parse_function

    original_parsed = parse_function(original_source := st.session_state["editor_original"])
    modified_parsed = parse_function(modified_source := st.session_state["editor_modified"])
    if original_parsed.parameters != modified_parsed.parameters:
        signature_html = (
            '<span class="bad">signature mismatch</span> · '
            f"({esc(', '.join(original_parsed.parameters))}) vs "
            f"({esc(', '.join(modified_parsed.parameters))})"
        )
    else:
        signature_html = (
            '<span class="ok">signatures compatible</span> · parameters: '
            f"{esc(', '.join(original_parsed.parameters))}"
        )
except EditError as exc:
    original_source = st.session_state["editor_original"]
    modified_source = st.session_state["editor_modified"]
    signature_html = f'<span class="bad">unsupported syntax</span> · {esc(str(exc)[:80])}'

_original_lines = (original_source or "").splitlines()
_modified_lines = (modified_source or "").splitlines()
_changed_line_count = sum(
    1
    for line in difflib.unified_diff(_original_lines, _modified_lines, lineterm="")
    if (line.startswith("+") and not line.startswith("+++"))
    or (line.startswith("-") and not line.startswith("---"))
)

pane_left, pane_right = st.columns(2, gap="small")

with pane_left:
    render(
        '<section class="rt-pane">'
        '<div class="rt-pane-head">'
        '<span class="rt-pane-label"><b>01</b> / Original implementation</span>'
        f'<span class="rt-pane-fn">{len(_original_lines)} lines · reviewed original</span>'
        "</div>"
    )
    original_source = st.text_area(
        "Original implementation source",
        key="editor_original",
        height=236,
        label_visibility="collapsed",
        help="The reviewed original. This exact text is what gets executed.",
    )
    render(
        '<div class="rt-pane-foot" style="border-top:0;border-radius:0 0 12px 12px">'
        f'<span class="rt-sig">{signature_html}</span>'
        "</div>"
    )

with pane_right:
    render(
        '<section class="rt-pane">'
        '<div class="rt-pane-head">'
        '<span class="rt-pane-label"><b>02</b> / Modified implementation</span>'
        f'<span class="rt-pane-fn">{len(_modified_lines)} lines · your edit</span>'
        "</div>"
    )
    modified_source = st.text_area(
        "Modified implementation source",
        key="editor_modified",
        height=236,
        label_visibility="collapsed",
        help="The proposed change. This exact text is what gets executed.",
    )
    render(
        '<div class="rt-pane-foot" style="border-top:0;border-radius:0 0 12px 12px">'
        f'<span class="rt-diffcount"><span class="n">{_changed_line_count}</span> '
        "line(s) differ</span>"
        '<span class="rt-sig">edited text is interpreted, never imported</span>'
        "</div>"
    )

# --- Primary action --------------------------------------------------------

action_left, action_right = st.columns([1.5, 1], gap="small")
with action_left:
    analyse_clicked = st.button(
        "Analyse change", type="primary", use_container_width=True
    )
with action_right:
    baseline_scenario = scenario_matching_sources(original_source, modified_source)
    baseline_clicked = st.button(
        "Run curated baseline",
        type="secondary",
        use_container_width=True,
        disabled=baseline_scenario is None,
        help=(
            "Runs the pre-written cases for the matching bundled example. "
            "Available only when the editors hold an unmodified example."
        ),
    )

if analyse_clicked:
    st.session_state["pending"] = "analyse"
if baseline_clicked and baseline_scenario is not None:
    st.session_state["pending"] = "baseline"
    st.session_state["baseline_scenario_id"] = baseline_scenario.id

render(
    '<div class="rt-control-note" style="margin-top:8px">'
    "<b>Analyse change</b> asks the local model for inputs, then executes both "
    "versions. <b>Curated baseline</b> runs pre-written cases, not the model."
    "</div>"
)

pending = st.session_state.get("pending")
st.session_state["pending"] = None

if pending == "analyse":
    perform_analysis(original_source, modified_source)
elif pending == "baseline":
    scenario = get_scenario(st.session_state.get("baseline_scenario_id", "discount"))
    st.session_state["failure"] = None
    st.session_state["run"] = pipeline.run_curated_baseline(scenario)

failure = st.session_state.get("failure")
run = st.session_state.get("run")

if run is not None and run.scenario_id not in {s.id for s in SCENARIOS.values()}:
    # A run for code that is no longer on screen; drop it to avoid stale evidence.
    st.session_state["run"] = None
    run = None

if failure:
    render_failure(failure)

if run is not None:
    render_run(run)
elif not failure:
    render(
        '<div class="rt-empty">'
        '<div class="rt-empty-title">No analysis yet.</div>'
        "<p>Load an example or paste two numeric functions, then press "
        "<b>Analyse change</b>. The local model proposes inputs and both versions "
        "are executed with identical arguments, so every outcome below comes from "
        "a real call.</p>"
        "</div>"
    )

# --- Footer ----------------------------------------------------------------

if run is not None and run.provenance == "curated":
    footer_input = "Curated baseline · not AI-generated"
    footer_method = "Pre-written inputs, executed by Python · not a model response"
else:
    footer_input = "Qwen2.5-Coder 7B via local Ollama"
    footer_method = "Model proposes data, never code · results are recorded, not generated"

render(
    '<div class="rt-footer">'
    f"<div><b>Test inputs</b> {esc(footer_input)} · Python executes both versions<br>"
    f'<span class="rt-mono">{esc(footer_method)}</span></div>'
    '<div class="rt-foot-right">Changed behaviour may be intentional. '
    "No difference found ≠ proof of correctness.</div>"
    "</div>"
)
