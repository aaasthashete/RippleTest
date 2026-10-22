"""RippleTest — AI-assisted behavioural regression discovery.

Single-page Streamlit workspace. The local model proposes test data; this file
only orchestrates, renders, and displays outcomes produced by real execution of
the bundled trusted functions.
"""

from __future__ import annotations

import difflib
import json
import re
from datetime import datetime
from html import escape
from typing import Any

import streamlit as st

from core import ollama_client, pipeline
from core.differential import CaseResult
from core.ollama_client import MODEL_NAME, ModelStatus
from core.report import run_report
from core.scenarios import SCENARIOS, Scenario, get_scenario
from core.validators import (
    ModelResponseError,
    RejectedCase,
    parse_json_content,
    validate_cases,
)

st.set_page_config(
    page_title="RippleTest — Behavioural regression discovery",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design system: Paper & Graphite
# ---------------------------------------------------------------------------

CSS = """
:root {
  --canvas: #F6F7F5;
  --surface: #FFFFFF;
  --surface-muted: #F0F2F1;
  --ink: #171C22;
  --ink-muted: #68727B;
  --line: #DDE2E3;
  --accent: #2E5BEF;
  --accent-deep: #2449C6;
  --accent-soft: #EBF0FF;
  --positive: #217451;
  --positive-soft: #E8F4EC;
  --danger: #B83F48;
  --danger-soft: #FBEDEF;
  --warning: #94611A;
  --warning-soft: #FFF3DF;
  --mono: "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
  --sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--canvas);
  color: var(--ink);
  font-family: var(--sans);
}
[data-testid="stAppViewContainer"] { font-size: 15px; }
[data-testid="stMainBlockContainer"] {
  max-width: 1380px;
  padding: 1.5rem 40px 5rem;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* Keep the editorial stack on every custom surface; code stays monospace. */
.rt-header, .rt-header *, .rt-eyebrow, .rt-title, .rt-lede,
.rt-contract, .rt-contract *, .rt-pane-head, .rt-pane-label, .rt-pane-fn,
.rt-section, .rt-section *, .rt-summary, .rt-summary *,
.rt-matrix-wrap, .rt-matrix-wrap *, .rt-inspector, .rt-inspector *,
.rt-notice, .rt-notice *, .rt-empty, .rt-empty *, .rt-footer, .rt-footer *,
.rt-detail-grid, .rt-detail-grid *, .rt-rejects, .rt-rejects *,
.rt-control-note, .rt-prov, .rt-chip {
  font-family: var(--sans);
}
.rt-header .rt-mark, .rt-pane-fn, .rt-pane-label, .rt-line, .rt-ln,
.rt-prov, .rt-prov-note, .rt-metric-value, .rt-matrix .num, .rt-matrix .mono,
.rt-exc, .rt-args, .rt-obs-value, .rt-obs-sub, .rt-delta, .rt-detail-value,
.rt-contract-bounds, .rt-contract-label, .rt-eyebrow, .rt-footer .rt-mono,
.rt-rejects li b, .rt-rejects code, .rt-notice-extra,
.rt-summary .rt-metric-value, .rt-status {
  font-family: var(--mono);
}

h1, h2, h3, h4, h5, h6 { font-family: var(--sans); color: var(--ink); }

code, pre, kbd { font-family: var(--mono); }

/* Header ---------------------------------------------------------------- */
.rt-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 20px; padding: 4px 0 16px; margin-bottom: 34px;
  border-bottom: 1px solid var(--line);
}
.rt-brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
.rt-mark {
  font-family: var(--mono); font-size: 13.5px; font-weight: 600; letter-spacing: -.02em;
  color: var(--surface); background: var(--ink); border-radius: 4px;
  padding: 3px 7px 4px; line-height: 1;
}
.rt-brand-name { font-size: 15.5px; font-weight: 640; letter-spacing: -.012em; }
.rt-brand-sub {
  font-size: 10px; font-weight: 600; letter-spacing: .13em; text-transform: uppercase;
  color: var(--ink-muted); border-left: 1px solid var(--line); padding-left: 12px;
}
.rt-status {
  display: inline-flex; align-items: center; gap: 8px; white-space: nowrap;
  font-family: var(--mono); font-size: 10.5px; letter-spacing: .07em;
  text-transform: uppercase; color: var(--ink-muted); cursor: default;
}
.rt-dot { width: 6px; height: 6px; border-radius: 50%; flex: none; }
.rt-dot--ok { background: var(--positive); }
.rt-dot--bad { background: var(--danger); }
.rt-dot--wait { background: var(--warning); }

/* Intro ----------------------------------------------------------------- */
.rt-eyebrow {
  font-family: var(--mono); font-size: 10px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--ink-muted); margin-bottom: 10px;
}
.rt-eyebrow b { color: var(--ink); font-weight: 600; }
.rt-eyebrow .sep { color: #B4BDC3; padding: 0 4px; }
.rt-title {
  font-family: var(--sans) !important;
  font-size: 32px !important; font-weight: 680 !important; letter-spacing: -.028em !important;
  line-height: 1.12 !important; margin: 0 0 10px !important; color: var(--ink) !important;
  padding: 0 !important;
}
h2.rt-section-title { font-family: var(--sans) !important; }
.rt-lede {
  font-size: 14.5px !important; line-height: 1.55 !important; color: var(--ink-muted) !important;
  max-width: 62ch; margin: 0 !important;
}
.rt-lede code, .rt-contract-text code {
  font-size: 12px; background: var(--surface-muted); color: #33404A;
  padding: 1px 4px; border-radius: 3px;
}
.rt-control-note {
  font-size: 11px; line-height: 1.45; color: var(--ink-muted); margin-top: 2px;
}
.rt-control-note b { color: #4C565E; font-weight: 600; }

/* Widgets --------------------------------------------------------------- */
[data-testid="stWidgetLabel"] p {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif !important;
  font-size: 10px !important; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ink-muted) !important; font-weight: 600 !important;
}
[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  min-height: 44px !important;
  font-size: 13.5px !important;
  color: var(--ink) !important;
}
[data-baseweb="select"] > div:hover { border-color: #C3CCD0 !important; }
[data-baseweb="select"] svg { color: var(--ink-muted); }
[data-testid="stBaseButton-primary"], button[kind="primary"] {
  background: var(--accent) !important; border: 1px solid var(--accent) !important;
  color: #FFFFFF !important; border-radius: 8px !important;
  font-size: 13.5px !important; font-weight: 600 !important;
  min-height: 44px !important; box-shadow: none !important;
}
[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover {
  background: var(--accent-deep) !important; border-color: var(--accent-deep) !important;
}
[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
  background: transparent !important; border: 1px solid var(--line) !important;
  color: var(--ink) !important; border-radius: 8px !important;
  font-size: 12.5px !important; font-weight: 550 !important;
  min-height: 38px !important; box-shadow: none !important;
}
[data-testid="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover {
  border-color: #C3CCD0 !important; background: var(--surface) !important;
}
button:focus-visible, [data-baseweb="select"]:focus-within {
  outline: 2px solid var(--accent) !important; outline-offset: 2px !important;
}
[data-testid="stCaptionContainer"] p, .stCaption p {
  font-size: 11.5px !important; color: var(--ink-muted) !important; line-height: 1.5;
}
[data-testid="stExpander"] {
  border: 1px solid var(--line) !important; border-radius: 8px !important;
  background: var(--surface) !important; box-shadow: none !important;
}
[data-testid="stExpander"] summary {
  font-size: 12.5px !important; font-weight: 550 !important; color: var(--ink) !important;
}
[data-testid="stExpander"] summary:hover { color: var(--accent) !important; }
[data-testid="stStatusWidget"] { font-size: 12px; }

/* Contract strip -------------------------------------------------------- */
.rt-contract {
  display: grid; grid-template-columns: 148px minmax(0, 1fr); gap: 18px;
  background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 18px; margin: 24px 0 26px;
}
.rt-contract-label {
  font-family: var(--mono); font-size: 10px; letter-spacing: .12em;
  text-transform: uppercase; color: var(--ink-muted); padding-top: 2px;
}
.rt-contract-text { font-size: 13.8px; line-height: 1.55; margin: 0; color: #26303A; }
.rt-contract-bounds {
  font-family: var(--mono); font-size: 11px; color: var(--ink-muted);
  margin-top: 6px; letter-spacing: .01em;
}

/* Code panes ------------------------------------------------------------ */
.rt-panes {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px; align-items: stretch;
}
.rt-pane {
  background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  overflow: hidden; display: flex; flex-direction: column; min-width: 0;
}
.rt-pane-head {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 9px 14px; border-bottom: 1px solid var(--line);
  background: #FBFCFB; min-height: 38px;
}
.rt-pane-label {
  font-family: var(--mono); font-size: 10px; letter-spacing: .1em;
  text-transform: uppercase; color: var(--ink-muted); white-space: nowrap;
}
.rt-pane-label b { color: var(--ink); font-weight: 600; }
.rt-pane-fn {
  font-family: var(--mono); font-size: 10.5px; color: #8C969D;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.rt-code { padding: 11px 0 12px; overflow-x: auto; flex: 1; }
.rt-line {
  display: flex; align-items: flex-start; min-height: 22px;
  font-family: var(--mono); font-size: 12.75px; line-height: 22px;
  white-space: pre; color: #20262E;
}
.rt-ln {
  width: 42px; flex: none; text-align: right; padding-right: 14px;
  color: #A8B1B8; font-size: 10.5px; user-select: none;
}
.rt-lc { padding-right: 20px; }
.rt-line--changed { background: var(--danger-soft); box-shadow: inset 2px 0 0 var(--danger); }
.rt-line--changed .rt-ln { color: #B07A80; }

/* Section head + provenance -------------------------------------------- */
.rt-section {
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 18px; margin: 34px 0 14px;
}
.rt-section-title {
  font-size: 17px !important; font-weight: 650 !important; letter-spacing: -.012em !important;
  margin: 0 !important; padding: 0 !important; color: var(--ink) !important;
}
.rt-section-sub { font-size: 12px; color: var(--ink-muted); margin-top: 4px; }
.rt-prov-row { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; justify-content: flex-end; }
.rt-prov {
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .09em; text-transform: uppercase;
  color: #44505A; background: var(--surface-muted); border: 1px solid var(--line);
  border-radius: 4px; padding: 3px 7px; white-space: nowrap;
}
.rt-prov--ai { color: var(--accent-deep); background: var(--accent-soft); border-color: #D6E0FF; }
.rt-prov-note { font-family: var(--mono); font-size: 10.5px; color: var(--ink-muted); }

/* Summary --------------------------------------------------------------- */
.rt-summary {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  overflow: hidden; margin-bottom: 16px;
}
.rt-metric { padding: 11px 18px 12px; border-left: 1px solid var(--line); }
.rt-metric:first-child { border-left: 0; }
.rt-metric-label {
  display: flex; align-items: center; gap: 6px;
  font-size: 9.5px; letter-spacing: .11em; text-transform: uppercase;
  color: var(--ink-muted); font-weight: 600; white-space: nowrap;
}
.rt-metric-value {
  font-family: var(--mono); font-size: 21px; font-weight: 500;
  font-variant-numeric: tabular-nums; line-height: 1.3; margin-top: 2px;
}
.rt-metric--changed .rt-metric-value { color: var(--danger); }
.rt-metric--error .rt-metric-value { color: var(--warning); }
.rt-flag { width: 5px; height: 5px; border-radius: 50%; flex: none; }
.rt-flag--danger { background: var(--danger); }
.rt-flag--warning { background: var(--warning); }

/* Matrix ---------------------------------------------------------------- */
.rt-matrix-wrap {
  border: 1px solid var(--line); border-radius: 10px; background: var(--surface);
  overflow: hidden;
}
.rt-matrix-scroll { overflow-x: auto; }
table.rt-matrix { width: 100%; border-collapse: collapse; min-width: 660px; }
.rt-matrix thead th {
  font-size: 9.5px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ink-muted); text-align: left; padding: 10px 16px;
  background: #FBFCFB; border-bottom: 1px solid var(--line); font-weight: 600;
  white-space: nowrap;
}
.rt-matrix td {
  padding: 0 16px; height: 48px; border-top: 1px solid var(--line);
  font-size: 13px; vertical-align: middle;
}
.rt-matrix tbody tr:first-child td { border-top: 0; }
.rt-matrix tbody tr:hover td { background: #F8FAF9; }
.rt-matrix .num {
  width: 48px; font-family: var(--mono); font-size: 11px; color: #97A1A8;
}
.rt-matrix .mono {
  font-family: var(--mono); font-size: 12.25px; font-variant-numeric: tabular-nums;
  color: #20262E; white-space: nowrap;
}
.rt-matrix .out { width: 150px; }
.rt-matrix .empty { color: #9AA4AB; }
.rt-row--changed td { background: var(--danger-soft); }
.rt-row--changed:hover td { background: #F8E6E9 !important; }
.rt-row--changed td:first-child { box-shadow: inset 2px 0 0 var(--danger); }
.rt-row--selected td { background: var(--accent-soft) !important; }
.rt-row--selected td:first-child { box-shadow: inset 2px 0 0 var(--accent); }
.rt-exc { color: var(--warning); font-family: var(--mono); font-size: 12px; }

.rt-chip {
  display: inline-flex; align-items: center; gap: 6px; border-radius: 5px;
  font-size: 11px; font-weight: 600; padding: 3px 8px; white-space: nowrap;
  border: 1px solid transparent;
}
.rt-chip .rt-dot { width: 5px; height: 5px; }
.rt-chip--same { color: var(--positive); background: var(--positive-soft); border-color: #D9EADD; }
.rt-chip--changed { color: var(--danger); background: var(--danger-soft); border-color: #F1D6DA; }
.rt-chip--error { color: var(--warning); background: var(--warning-soft); border-color: #F0DFC0; }
.rt-matrix .rt-chip--changed { background: #FFFFFF; }

/* Inspector ------------------------------------------------------------- */
.rt-inspector {
  border: 1px solid var(--line); border-radius: 10px; background: var(--surface);
  padding: 17px 20px 18px; margin-top: 10px;
}
.rt-inspector-head {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 16px; border-bottom: 1px solid var(--line);
  padding-bottom: 12px; margin-bottom: 15px;
}
.rt-inspector-title { font-size: 14.5px; font-weight: 650; letter-spacing: -.008em; }
.rt-inspector-title span { color: var(--ink-muted); font-weight: 500; }
.rt-field-label {
  font-size: 9.5px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ink-muted); font-weight: 600; margin-bottom: 6px;
}
.rt-args {
  font-family: var(--mono); font-size: 12.8px; color: #20262E;
  background: var(--surface-muted); border-radius: 6px; padding: 9px 12px;
  margin-bottom: 14px; overflow-x: auto; white-space: nowrap;
}
.rt-args b { font-weight: 500; color: var(--ink-muted); }
.rt-grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.rt-obs { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px 11px; }
.rt-obs--changed { border-color: #ECCED3; }
.rt-obs--error { border-color: #EDDCBC; background: #FFFCF6; }
.rt-obs-value {
  font-family: var(--mono); font-size: 15.5px; font-variant-numeric: tabular-nums;
  color: var(--ink); word-break: break-word;
}
.rt-obs--error .rt-obs-value { color: var(--warning); font-size: 13.5px; }
.rt-obs-sub { font-family: var(--mono); font-size: 10.5px; color: var(--ink-muted); margin-top: 3px; }
.rt-delta {
  font-family: var(--mono); font-size: 11.5px; color: var(--ink-muted);
  margin-top: 10px;
}
.rt-delta b { color: var(--ink); font-weight: 500; }
.rt-rationale { margin-top: 15px; padding-top: 13px; border-top: 1px solid var(--line); }
.rt-rationale p { margin: 0; font-size: 13.2px; line-height: 1.55; color: #26303A; }
.rt-subnote { font-size: 11px; color: var(--ink-muted); margin-top: 5px; display: block; }
.rt-note { font-size: 12px; color: var(--warning); margin-top: 9px; }

/* Notices, empty, details ----------------------------------------------- */
.rt-notice {
  border: 1px solid var(--line); border-left: 3px solid var(--accent);
  background: var(--surface); border-radius: 8px; padding: 13px 17px;
  margin: 0 0 16px;
}
.rt-notice--error, .rt-notice--changed { border-left-color: var(--danger); }
.rt-notice--warning { border-left-color: var(--warning); }
.rt-notice-label {
  font-size: 9.5px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--ink-muted); font-weight: 600; margin-bottom: 5px;
}
.rt-notice p { margin: 0; font-size: 13.2px; line-height: 1.5; color: #26303A; }
.rt-notice .rt-notice-extra {
  margin-top: 6px; font-family: var(--mono); font-size: 11px;
  color: var(--ink-muted); line-height: 1.5;
}
.rt-empty {
  border: 1px solid var(--line); border-radius: 10px; background: var(--surface);
  padding: 26px 24px; margin-top: 10px;
}
.rt-empty-title { font-size: 14.5px; font-weight: 600; margin-bottom: 6px; }
.rt-empty p { margin: 0; font-size: 13.2px; line-height: 1.55; color: var(--ink-muted); max-width: 78ch; }

.rt-detail-grid {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px 20px; margin-bottom: 4px;
}
.rt-detail-label {
  font-size: 9.5px; letter-spacing: .11em; text-transform: uppercase;
  color: var(--ink-muted); font-weight: 600; margin-bottom: 3px;
}
.rt-detail-value { font-family: var(--mono); font-size: 12px; color: var(--ink); }
.rt-rejects { margin: 10px 0 0; padding: 0; list-style: none; }
.rt-rejects li {
  border-top: 1px solid var(--line); padding: 8px 0;
  font-size: 12px; color: #3A454E; line-height: 1.5;
}
.rt-rejects li b { font-family: var(--mono); font-weight: 500; color: var(--ink-muted); }
.rt-rejects code {
  display: block; font-family: var(--mono); font-size: 10.5px;
  color: #8C969D; margin-top: 2px; overflow-wrap: anywhere;
}

/* Footer ---------------------------------------------------------------- */
.rt-footer {
  display: flex; justify-content: space-between; align-items: flex-start; gap: 24px;
  margin-top: 38px; padding-top: 15px; border-top: 1px solid var(--line);
  font-size: 11px; line-height: 1.55; color: var(--ink-muted);
}
.rt-footer b { color: #44505A; font-weight: 600; }
.rt-footer .rt-foot-right { text-align: right; }
.rt-footer .rt-mono { font-family: var(--mono); font-size: 10.5px; }

/* Responsive ------------------------------------------------------------ */
@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"] { padding: 1rem 18px 3.5rem; }
  .rt-header { flex-direction: column; align-items: flex-start; gap: 10px; margin-bottom: 24px; }
  .rt-brand-sub { display: none; }
  .rt-title { font-size: 27px !important; }
  .rt-panes, .rt-grid-2 { grid-template-columns: minmax(0, 1fr); }
  .rt-contract { grid-template-columns: minmax(0, 1fr); gap: 7px; }
  .rt-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .rt-metric:nth-child(3) { border-left: 0; border-top: 1px solid var(--line); }
  .rt-metric:nth-child(4) { border-top: 1px solid var(--line); }
  .rt-section { flex-direction: column; align-items: flex-start; gap: 8px; }
  .rt-prov-row { justify-content: flex-start; }
  .rt-footer { flex-direction: column; gap: 10px; }
  .rt-footer .rt-foot-right { text-align: left; }
  .rt-detail-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------


def render(fragment: str) -> None:
    st.markdown(fragment, unsafe_allow_html=True)


def esc(value: object) -> str:
    return escape(str(value), quote=True)


def inline_code(text: str) -> str:
    """Escape text, then turn `backticked` spans into inline <code>."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", esc(text))


def format_argument(value: Any) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


@st.cache_data(ttl=8, show_spinner=False)
def read_model_status() -> ModelStatus:
    """Cached health check; short TTL so the header reflects reality."""
    return ollama_client.check_model()


def reset_results() -> None:
    st.session_state["run"] = None
    st.session_state["failure"] = None
    st.session_state["pending"] = None
    st.session_state["generation_notice"] = ""


def header_markup(status: ModelStatus | None = None) -> str:
    """status=None renders the honest pre-flight state while /api/tags is called."""
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
        f'<span class="rt-brand-sub">Differential testing</span></div>'
        f'<span class="rt-status" title="{esc(detail)}">'
        f'<span class="rt-dot {dot}"></span>{esc(text)}</span>'
        f"</div>"
    )


def notice_markup(
    label: str,
    message: str,
    *,
    variant: str = "info",
    extra: str = "",
) -> str:
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
            f"Run <code>ollama pull {esc(MODEL_NAME)}</code>, then check the connection again.",
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
# Code panes
# ---------------------------------------------------------------------------


def changed_lines(left: str, right: str) -> tuple[set[int], set[int]]:
    left_lines = left.rstrip("\n").split("\n")
    right_lines = right.rstrip("\n").split("\n")
    matcher = difflib.SequenceMatcher(a=left_lines, b=right_lines, autojunk=False)
    left_changed: set[int] = set()
    right_changed: set[int] = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            left_changed.update(range(i1, i2))
        if tag in ("replace", "insert"):
            right_changed.update(range(j1, j2))
    return left_changed, right_changed


def code_pane(
    number: str,
    label: str,
    function_name: str,
    source: str,
    changed: set[int],
) -> str:
    lines = source.rstrip("\n").split("\n")
    rendered_lines = []
    for index, line in enumerate(lines):
        css = "rt-line rt-line--changed" if index in changed else "rt-line"
        rendered_lines.append(
            f'<div class="{css}"><span class="rt-ln">{index + 1}</span>'
            f'<span class="rt-lc">{esc(line) if line else " "}</span></div>'
        )
    return (
        f'<section class="rt-pane">'
        f'<div class="rt-pane-head">'
        f'<span class="rt-pane-label"><b>{esc(number)}</b> / {esc(label)}</span>'
        f'<span class="rt-pane-fn">{esc(function_name)} · read-only</span>'
        f"</div>"
        f'<div class="rt-code">{"".join(rendered_lines)}</div>'
        f"</section>"
    )


def code_panes(scenario: Scenario) -> None:
    left_changed, right_changed = changed_lines(scenario.original_source, scenario.modified_source)
    render(
        '<div class="rt-panes">'
        + code_pane("01", "Original", scenario.function_name, scenario.original_source, left_changed)
        + code_pane("02", "Modified", scenario.function_name, scenario.modified_source, right_changed)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


CHIPS = {
    "same": ("rt-chip--same", "Same result"),
    "changed": ("rt-chip--changed", "Changed"),
    "error": ("rt-chip--error", "Error"),
}
OUTCOME_ORDER = {"changed": 0, "error": 1, "same": 2}


def chip(classification: str) -> str:
    tone, label = CHIPS[classification]
    return f'<span class="rt-chip {tone}"><span class="rt-dot"></span>{label}</span>'


def outcome_cell(result: CaseResult, outcome) -> str:
    if outcome.kind == "exception":
        return f'<span class="rt-exc">{esc(outcome.exception_type)}</span>'
    return esc(outcome.display)


def matrix_markup(results: tuple[CaseResult, ...], selected_number: int) -> str:
    ordered = sorted(results, key=lambda item: (OUTCOME_ORDER[item.classification], item.number))
    rows = []
    for result in ordered:
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
            f'<td class="mono">{outcome_cell(result, result.original)}</td>'
            f'<td class="mono">{outcome_cell(result, result.modified)}</td>'
            f'<td class="out">{chip(result.classification)}</td>'
            f"</tr>"
        )
    return (
        '<div class="rt-matrix-wrap"><div class="rt-matrix-scroll">'
        '<table class="rt-matrix"><thead><tr>'
        '<th scope="col">#</th><th scope="col">Input</th>'
        '<th scope="col">Original</th><th scope="col">Modified</th>'
        '<th scope="col">Outcome</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div></div>"
    )


def summary_markup(summary) -> str:
    metrics = [
        ("Executed", summary.executed, ""),
        ("Same outcome", summary.same, ""),
        ("Behaviour changed", summary.changed, "changed"),
        ("Execution errors", summary.errors, "error"),
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


def inspector_markup(scenario: Scenario, result: CaseResult) -> str:
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
                f'<div class="rt-obs-sub">{esc(note)}</div>'
                f"</div>"
            )
        return (
            f'<div class="rt-obs">'
            f'<div class="rt-field-label">{esc(label)}</div>'
            f'<div class="rt-obs-value">{esc(outcome.display)}</div>'
            f'<div class="rt-obs-sub">{esc(type(outcome.value).__name__)}</div>'
            f"</div>"
        )

    delta = ""
    if result.delta is not None:
        delta = (
            f'<div class="rt-delta">modified − original = '
            f"<b>{result.delta:+.12g}</b></div>"
        )

    if result.provenance == "ai":
        reason_label = "Why the model proposed it"
        subnote = "Model suggestion, not execution evidence. The outputs above come from Python."
    else:
        reason_label = "Why this baseline case exists"
        subnote = "Pre-written case. This run did not use the model."

    note = f'<div class="rt-note">{esc(result.note)}</div>' if result.note else ""

    return (
        '<div class="rt-inspector">'
        f'<div class="rt-inspector-head">'
        f'<div class="rt-inspector-title">Case {result.number:02d} '
        f'<span>/ {esc(scenario.function_name)}</span></div>'
        f"{chip(result.classification)}</div>"
        f'<div class="rt-field-label">Identical arguments passed to both versions</div>'
        f'<div class="rt-args">{arguments}</div>'
        f'<div class="rt-grid-2">'
        f'{observation("01 / Original result", result.original)}'
        f'{observation("02 / Modified result", result.modified)}'
        f"</div>"
        f"{delta}{note}"
        f'<div class="rt-rationale">'
        f'<div class="rt-field-label">{reason_label}</div>'
        f"<p>{esc(result.reason)}</p>"
        f'<span class="rt-subnote">{subnote}</span>'
        f"</div></div>"
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
        schema_note = "—"
        if run.provenance == "ai":
            schema_note = "Numeric arguments validated"
        render(
            '<div class="rt-detail-grid">'
            f'<div><div class="rt-detail-label">Input source</div>'
            f'<div class="rt-detail-value">'
            f'{"AI-generated" if run.provenance == "ai" else "Curated baseline"}</div></div>'
            f'<div><div class="rt-detail-label">Model</div>'
            f'<div class="rt-detail-value">{esc(run.model or "none (curated)")}</div></div>'
            f'<div><div class="rt-detail-label">Request time</div>'
            f'<div class="rt-detail-value">{esc(request_time)}</div></div>'
            f'<div><div class="rt-detail-label">Comparison tolerance</div>'
            f'<div class="rt-detail-value">rel 1e-9 · abs 1e-9</div></div>'
            f'<div><div class="rt-detail-label">Proposed</div>'
            f'<div class="rt-detail-value">{run.validation.proposed_count}</div></div>'
            f'<div><div class="rt-detail-label">Accepted</div>'
            f'<div class="rt-detail-value">{len(run.validation.accepted)}</div></div>'
            f'<div><div class="rt-detail-label">Rejected</div>'
            f'<div class="rt-detail-value">{len(run.validation.rejected)}</div></div>'
            f'<div><div class="rt-detail-label">Input handling</div>'
            f'<div class="rt-detail-value">{esc(schema_note)}</div></div>'
            f"</div>"
        )
        if run.validation.rejected:
            render(
                '<div class="rt-field-label" style="margin-top:14px">Rejected proposals</div>'
                + rejected_markup(run.validation.rejected)
            )
        else:
            st.caption("Every proposed case passed validation and was executed.")
        st.caption(
            "Only bundled, pre-reviewed functions were called. The model supplies numeric "
            "arguments only — never executable code."
        )
        st.download_button(
            "Download JSON report",
            data=json.dumps(run_report(run), indent=2, ensure_ascii=False),
            file_name=f"rippletest-{run.scenario_id}-{run.run_id}.json",
            mime="application/json",
            type="secondary",
        )


def render_run(scenario: Scenario, run) -> None:
    summary = run.summary
    provenance = (
        f'<span class="rt-prov rt-prov--ai">AI-generated inputs</span>'
        f'<span class="rt-prov-note">{esc(run.model or MODEL_NAME)} · local Ollama</span>'
        if run.provenance == "ai"
        else '<span class="rt-prov">Curated baseline</span>'
        '<span class="rt-prov-note">pre-written cases · not AI-generated</span>'
    )
    stamp = datetime.fromisoformat(run.created_at).astimezone().strftime("%H:%M")

    render(
        f'<div class="rt-section">'
        f'<div><h2 class="rt-section-title">Observed behaviour</h2>'
        f'<div class="rt-section-sub">{summary.executed} cases executed at {stamp}. '
        f"Outcomes come from real calls to both versions.</div></div>"
        f'<div class="rt-prov-row">{provenance}</div></div>'
    )
    render(summary_markup(summary))

    ordered = sorted(
        run.results, key=lambda item: (OUTCOME_ORDER[item.classification], item.number)
    )
    options = [item.number for item in ordered]
    key = f"inspect_{run.run_id}"
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    selected_number = st.session_state[key]

    if summary.changed:
        render(
            notice_markup(
                "Behaviour changed",
                f"{summary.changed} of {summary.executed} executed cases returned a different "
                "outcome. Review the evidence below; a difference is not automatically a bug.",
                variant="changed",
            )
        )
    else:
        render(
            notice_markup(
                "No behavioural differences observed",
                f"All {summary.executed} executed cases produced matching outcomes. "
                "This does not establish that the change is regression-free.",
            )
        )

    render(matrix_markup(run.results, selected_number))
    st.caption("Sorted by outcome; case numbers are stable. Select a case to inspect its evidence.")

    head_left, head_right = st.columns([2.2, 1], gap="large")
    with head_left:
        render(
            '<h2 class="rt-section-title" style="margin-top:18px">Case evidence</h2>'
        )
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
    render(inspector_markup(scenario, selected))
    run_details(run)


def render_failure(failure: dict[str, Any]) -> None:
    render(
        notice_markup(
            failure.get("title", "Run failed"),
            failure.get("message", "The run could not be completed."),
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
# Actions that make real requests and execute trusted code
# ---------------------------------------------------------------------------


def perform_generation(scenario: Scenario) -> None:
    st.session_state["failure"] = None
    st.session_state["run"] = None
    st.session_state["generation_notice"] = ""

    with st.status("Generating test inputs", expanded=True) as box:
        st.write(f"Requesting {MODEL_NAME} from local Ollama")
        generation = ollama_client.generate_test_cases(scenario)

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

        if generation.notice:
            st.write(generation.notice)
            st.session_state["generation_notice"] = generation.notice

        st.write("Validating proposed inputs")
        try:
            payload = parse_json_content(generation.raw_content)
            validation = validate_cases(scenario, payload)
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
                    "validation. Nothing was executed. Generate again, or try the curated baseline."
                ),
                "detail": "",
                "raw": generation.raw_content,
                "rejected": validation.rejected,
            }
            return

        if len(validation.accepted) < 5:
            st.write(
                f"Only {len(validation.accepted)} of {validation.proposed_count} proposals were usable."
            )

        st.write(f"Executing both versions on {len(validation.accepted)} accepted cases")
        run = pipeline.build_run(
            scenario,
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

for state_key in ("run", "failure", "generation_notice"):
    if state_key not in st.session_state:
        st.session_state[state_key] = None if state_key != "generation_notice" else ""

header_slot = st.empty()
header_slot.markdown(header_markup(), unsafe_allow_html=True)
status = read_model_status()
header_slot.markdown(header_markup(status), unsafe_allow_html=True)

intro_col, controls_col = st.columns([1.65, 1], gap="large")

with intro_col:
    render(
        '<div class="rt-eyebrow"><b>Workspace 01</b>'
        '<span class="sep">/</span>Regression analysis</div>'
        '<h1 class="rt-title">Find the break before it ships.</h1>'
        '<p class="rt-lede">Compare two versions of a trusted Python function. '
        "Qwen2.5-Coder 7B proposes the test inputs on this Mac; Python runs both "
        "versions and shows exactly what changed.</p>"
    )

with controls_col:
    scenario_id = st.selectbox(
        "Scenario",
        options=list(SCENARIOS),
        format_func=lambda key: SCENARIOS[key].label,
        key="scenario_id",
        on_change=reset_results,
    )
    generate_clicked = st.button("Generate & run tests", type="primary", width="stretch")
    baseline_clicked = st.button("Run curated baseline", type="secondary", width="stretch")
    render(
        '<div class="rt-control-note"><b>Curated baseline</b> uses pre-written cases, '
        "not the model.</div>"
    )

scenario = get_scenario(scenario_id)

if generate_clicked:
    st.session_state["pending"] = "generate"
if baseline_clicked:
    st.session_state["pending"] = "baseline"

if status.state != "available":
    render(status_notice(status))
    if st.button("Check connection", type="secondary", key="recheck_top"):
        read_model_status.clear()
        st.rerun()

render(
    '<div class="rt-contract">'
    '<div class="rt-contract-label">Intended behaviour</div>'
    f'<div><p class="rt-contract-text">{inline_code(scenario.contract)}</p>'
    f'<div class="rt-contract-bounds">Inputs — {esc(scenario.constraints_text)} '
    f"· finite numbers only · {len(scenario.parameters)} arguments</div></div>"
    "</div>"
)

code_panes(scenario)

pending = st.session_state.get("pending")
st.session_state["pending"] = None

if pending == "generate":
    perform_generation(scenario)
elif pending == "baseline":
    st.session_state["failure"] = None
    st.session_state["generation_notice"] = ""
    st.session_state["run"] = pipeline.run_curated_baseline(scenario)

failure = st.session_state.get("failure")
run = st.session_state.get("run")

if run is not None and run.scenario_id != scenario.id:
    st.session_state["run"] = None
    run = None

if failure:
    render_failure(failure)

if run is not None:
    if st.session_state.get("generation_notice"):
        render(
            notice_markup(
                "Model transport note",
                esc(st.session_state["generation_notice"]),
                variant="warning",
            )
        )
    render_run(scenario, run)
elif not failure:
    render(
        '<div class="rt-empty">'
        '<div class="rt-empty-title">No test run yet.</div>'
        "<p>Generate inputs with the local model, or run the curated baseline. "
        "Either way, both versions above receive the same arguments and every "
        "outcome shown here will come from an actual call.</p>"
        "</div>"
    )

if run is not None and run.provenance == "curated":
    footer_input = "Curated baseline · not AI-generated"
    footer_method = "Pre-written inputs, executed by Python · not a model response"
else:
    footer_input = "Qwen2.5-Coder 7B via local Ollama"
    footer_method = "Model proposes data, never code · results are recorded, not generated"

render(
    '<div class="rt-footer">'
    f"<div><b>Test inputs</b> {esc(footer_input)} · Python executes both versions<br>"
    f'<span class="rt-mono">{esc(footer_method)}</span>'
    "</div>"
    '<div class="rt-foot-right">Changed behaviour may be intentional. '
    "No difference found ≠ proof of correctness.</div>"
    "</div>"
)
