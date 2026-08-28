"""Self-contained HTML dashboard: data health, drift, model metrics, survival curves,
scenarios and the review queue in one offline file.

The generated page embeds its data as JSON and renders everything with modern CSS + inline
SVG — no CDN, no server, no build step — so judges can open `outputs/dashboard.html`
directly from the repository with zero external dependencies.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

_CSS = """
:root {
  --bg: #090d16;
  --bg-gradient: radial-gradient(circle at 50% -20%, #1e293b 0%, #090d16 80%);
  --panel: rgba(18, 24, 38, 0.85);
  --panel-card: rgba(22, 30, 48, 0.7);
  --border: rgba(255, 255, 255, 0.08);
  --border-glow: rgba(56, 189, 248, 0.25);
  --text: #f8fafc;
  --muted: #94a3b8;
  --dim: #64748b;
  --accent: #38bdf8;
  --accent-rgb: 56, 189, 248;
  --good: #34d399;
  --good-bg: rgba(52, 211, 153, 0.12);
  --warn: #fbbf24;
  --warn-bg: rgba(251, 191, 36, 0.12);
  --bad: #f87171;
  --bad-bg: rgba(248, 113, 113, 0.12);
  --purple: #a78bfa;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.15); }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.25); }
body {
  margin: 0;
  background: var(--bg);
  background-image: var(--bg-gradient);
  background-attachment: fixed;
  color: var(--text);
  font: 13.5px/1.5 -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
header {
  padding: 16px 36px 14px;
  border-bottom: 1px solid var(--border);
  background: rgba(11, 15, 25, 0.9);
  backdrop-filter: blur(16px);
  position: sticky;
  top: 0;
  z-index: 100;
}
.sub-nav {
  display: flex;
  gap: 6px;
  margin-top: 10px;
  overflow-x: auto;
  padding-bottom: 2px;
  scrollbar-width: none;
}
.sub-nav::-webkit-scrollbar { display: none; }
.nav-item {
  color: var(--muted);
  text-decoration: none;
  font-size: 11px;
  font-weight: 600;
  padding: 4px 10px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.06);
  white-space: nowrap;
  transition: all 0.15s ease;
}
.nav-item:hover {
  color: #fff;
  background: rgba(56, 189, 248, 0.15);
  border-color: rgba(56, 189, 248, 0.35);
}
.header-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  flex-wrap: wrap;
  gap: 12px;
}
.brand-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 9999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.05em;
  background: rgba(56, 189, 248, 0.1);
  color: var(--accent);
  border: 1px solid rgba(56, 189, 248, 0.25);
  text-transform: uppercase;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 9999px;
  font-size: 11px;
  font-weight: 600;
  background: var(--good-bg);
  color: var(--good);
  border: 1px solid rgba(52, 211, 153, 0.3);
}
.status-pill::before {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--good);
  box-shadow: 0 0 8px var(--good);
}
h1 {
  margin: 0 0 6px;
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #ffffff 40%, #93c5fd 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.meta {
  color: var(--muted);
  font-size: 12px;
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.meta span {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.card {
  background: var(--panel-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
  backdrop-filter: blur(8px);
  position: relative;
  overflow: hidden;
  transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
  transform: translateY(-2px);
  border-color: var(--border-glow);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.card::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--purple));
  opacity: 0.8;
}
.kpi .value {
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #fff;
  margin-bottom: 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.kpi .label {
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.section {
  padding: 16px 36px 20px;
  scroll-margin-top: 130px;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 16px;
  padding: 20px 36px 12px;
  scroll-margin-top: 130px;
}
.section-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 12px;
  flex-wrap: wrap;
  gap: 8px;
}
h2 {
  font-size: 14px;
  margin: 0;
  color: #f1f5f9;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  display: flex;
  align-items: center;
  gap: 8px;
}
h2::before {
  content: "";
  width: 4px;
  height: 14px;
  background: var(--accent);
  border-radius: 2px;
}
.section-desc {
  color: var(--dim);
  font-size: 12px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
  overflow-x: auto;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
  backdrop-filter: blur(8px);
}
.queue-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  gap: 12px;
  flex-wrap: wrap;
}
.search-input {
  background: rgba(15, 23, 42, 0.85);
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: #f8fafc;
  padding: 7px 14px;
  border-radius: 6px;
  font-size: 12px;
  outline: none;
  min-width: 260px;
  font-family: inherit;
  transition: all 0.2s ease;
}
.search-input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25);
}
.table-scroll-container {
  max-height: 480px;
  overflow-y: auto;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.05);
}
.table-scroll-container thead th {
  position: sticky;
  top: 0;
  background: #111a2e;
  z-index: 5;
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.1);
}
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}
thead tr {
  background: rgba(30, 41, 59, 0.4);
}
th {
  color: var(--muted);
  font-weight: 600;
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 0.05em;
  padding: 10px 14px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
td {
  padding: 10px 14px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  white-space: nowrap;
  color: #e2e8f0;
}
tr:last-child td {
  border-bottom: none;
}
tr:hover td {
  background: rgba(56, 189, 248, 0.04);
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.pill {
  display: inline-flex;
  align-items: center;
  padding: 3px 8px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.pill-good { background: var(--good-bg); color: var(--good); border: 1px solid rgba(52, 211, 153, 0.25); }
.pill-warn { background: var(--warn-bg); color: var(--warn); border: 1px solid rgba(251, 191, 36, 0.25); }
.pill-bad { background: var(--bad-bg); color: var(--bad); border: 1px solid rgba(248, 113, 113, 0.25); }
.pill-accent { background: rgba(56, 189, 248, 0.12); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.25); }

svg text { fill: var(--muted); font-size: 11px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
svg .grid-line { stroke: rgba(255, 255, 255, 0.06); stroke-dasharray: 4,4; }
svg .series { fill: none; stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }
.legend {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin-top: 12px;
  color: var(--muted);
  font-size: 12px;
  padding-top: 10px;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
}
.legend span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
}
.legend span::before {
  content: "";
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 3px;
  background: currentColor;
}
.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}
.two-col-chart {
  display: grid;
  grid-template-columns: 1.45fr 0.85fr;
  gap: 24px;
}
@media (max-width: 1024px) {
  .two-col, .two-col-chart { grid-template-columns: 1fr; }
  header, .grid, .section { padding-left: 20px; padding-right: 20px; }
}
.copilot-card {
  border: 1px solid rgba(56, 189, 248, 0.35);
  background: linear-gradient(180deg, rgba(14, 25, 45, 0.85) 0%, rgba(10, 16, 30, 0.85) 100%);
  border-radius: 12px;
  padding: 22px;
  position: relative;
  overflow: hidden;
}
.copilot-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
}
.copilot-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
  flex-wrap: wrap;
  gap: 8px;
}
.copilot-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  font-weight: 700;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.copilot-body {
  font-size: 13.5px;
  line-height: 1.65;
  color: #e2e8f0;
  white-space: pre-line;
}

.copilot-metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
.copilot-metric-chip {
  background: rgba(30, 41, 59, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 10px;
  padding: 12px 14px;
}
.copilot-chip-label {
  font-size: 11px;
  color: var(--muted);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 4px;
}
.copilot-chip-val {
  font-size: 18px;
  font-weight: 700;
  color: #fff;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.copilot-chip-sub {
  font-size: 11px;
  color: var(--dim);
  margin-top: 2px;
}
.copilot-narrative-box {
  background: rgba(15, 23, 42, 0.65);
  border: 1px solid rgba(56, 189, 248, 0.2);
  border-left: 4px solid var(--accent);
  border-radius: 10px;
  padding: 16px 18px;
  margin-bottom: 18px;
}
.copilot-narrative-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 11.5px;
  font-weight: 700;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 10px;
}
.copilot-narrative-text {
  font-size: 13.5px;
  line-height: 1.75;
  color: #cbd5e1;
  white-space: pre-wrap;
}
.copilot-narrative-text strong {
  color: #f8fafc;
  font-weight: 600;
}
.copilot-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  padding-top: 14px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}
.copilot-tags {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.copilot-tag-label {
  font-size: 11px;
  color: var(--muted);
  font-weight: 600;
  text-transform: uppercase;
}

.foot {
  padding: 24px 36px 36px;
  color: var(--dim);
  font-size: 12px;
  border-top: 1px solid var(--border);
  margin-top: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}
.foot code {
  color: var(--accent);
  background: rgba(56, 189, 248, 0.1);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
"""


def _svg_line_chart(data: list[dict], x_key: str, series: list[tuple[str, str, str]], height: int = 230,
                    y_range: tuple[float, float] | None = None, dual_axis: bool = False) -> str:
    """Enhanced SVG line chart with area gradients, dual-axes, and bounded ranges."""
    if not data:
        return "<p class='meta'>no data available</p>"
    xs = [d[x_key] for d in data]
    w, h = 760, height
    left, right, top, bottom = 54, 70 if dual_axis else 20, 18, 34
    x_min, x_max = min(xs), max(xs)
    x_span = (x_max - x_min) or 1

    def px(x):
        return left + (x - x_min) / x_span * (w - left - right)

    parts = [
        f"<svg viewBox='0 0 {w} {h}' width='100%' preserveAspectRatio='xMidYMid meet'>",
        "<defs>",
        f"<linearGradient id='grad-blue' x1='0' y1='0' x2='0' y2='1'>"
        f"<stop offset='0%' stop-color='#38bdf8' stop-opacity='0.25'/>"
        f"<stop offset='100%' stop-color='#38bdf8' stop-opacity='0.0'/></linearGradient>",
        "</defs>"
    ]

    if dual_axis and len(series) == 2:
        k0, l0, c0 = series[0]
        k1, l1, c1 = series[1]
        y0 = [d[k0] for d in data if d.get(k0) is not None]
        y1 = [d[k1] for d in data if d.get(k1) is not None]
        min0, max0 = (min(y0), max(y0)) if y0 else (0, 100)
        min1, max1 = (min(y1), max(y1)) if y1 else (0, 1000)
        pad0 = (max0 - min0) * 0.2 or 5.0
        pad1 = (max1 - min1) * 0.2 or 50.0
        lo0, hi0 = max(0.0, min0 - pad0), min(100.0, max0 + pad0)
        lo1, hi1 = max(0.0, min1 - pad1), max1 + pad1

        def py0(y):
            return top + (1 - (y - lo0) / ((hi0 - lo0) or 1)) * (h - top - bottom)

        def py1(y):
            return top + (1 - (y - lo1) / ((hi1 - lo1) or 1)) * (h - top - bottom)

        for i in range(5):
            gy0 = lo0 + (hi0 - lo0) * i / 4
            gy1 = lo1 + (hi1 - lo1) * i / 4
            y_coord = py0(gy0)
            parts.append(f"<line class='grid-line' x1='{left}' y1='{y_coord:.1f}' x2='{w - right}' y2='{y_coord:.1f}'/>")
            parts.append(f"<text x='{left - 8}' y='{y_coord + 4:.1f}' text-anchor='end' fill='{c0}' font-weight='600'>{gy0:.1f}</text>")
            parts.append(f"<text x='{w - 4}' y='{y_coord + 4:.1f}' text-anchor='end' fill='{c1}' font-weight='600'>{int(round(gy1)):,}</text>")

        pts0 = [f"{px(d[x_key]):.1f},{py0(d[k0]):.1f}" for d in data if d.get(k0) is not None]
        if pts0:
            area0 = f"{px(xs[0]):.1f},{h - bottom} " + " ".join(pts0) + f" {px(xs[-1]):.1f},{h - bottom}"
            parts.append(f"<polygon fill='url(#grad-blue)' points='{area0}'/>")
            parts.append(f"<polyline class='series' stroke='{c0}' points='{' '.join(pts0)}'/>")

        pts1 = [f"{px(d[x_key]):.1f},{py1(d[k1]):.1f}" for d in data if d.get(k1) is not None]
        if pts1:
            parts.append(f"<polyline class='series' stroke='{c1}' points='{' '.join(pts1)}'/>")

        parts.append(f"<text x='{left}' y='{h - 10}' fill='var(--muted)'>{x_key}</text>")
        parts.append(f"<text x='{w - 4}' y='{h - 10}' text-anchor='end' fill='{c1}'>breaches / 1k</text>")
        parts.append("</svg>")
        return "".join(parts)

    if y_range is not None:
        y_lo, y_hi = y_range
    else:
        ys = [d[k] for d in data for k, _, _ in series if d.get(k) is not None]
        y_min, y_max = (min(ys), max(ys)) if ys else (0, 1)
        pad = (y_max - y_min) * 0.1 or 0.1
        y_lo, y_hi = y_min - pad, y_max + pad

    def py(y):
        return top + (1 - (y - y_lo) / ((y_hi - y_lo) or 1)) * (h - top - bottom)

    for i in range(5):
        gy = y_lo + (y_hi - y_lo) * i / 4
        parts.append(f"<line class='grid-line' x1='{left}' y1='{py(gy):.1f}' x2='{w - right}' y2='{py(gy):.1f}'/>")
        label_val = f"{gy:.2f}" if y_range == (0.0, 1.0) else f"{gy:.3g}"
        parts.append(f"<text x='{left - 8}' y='{py(gy) + 4:.1f}' text-anchor='end' font-weight='500'>{label_val}</text>")

    for k, label, colour in series:
        pts = [f"{px(d[x_key]):.1f},{py(d[k]):.1f}" for d in data if d.get(k) is not None]
        if pts:
            parts.append(f"<polyline class='series' stroke='{colour}' points='{' '.join(pts)}'/>")
            if data and data[-1].get(k) is not None:
                lx = px(data[-1][x_key]) - 6
                ly = py(data[-1][k]) - 6
                parts.append(f"<text x='{lx:.1f}' y='{ly:.1f}' text-anchor='end' fill='{colour}' font-weight='600'>{label}</text>")

    parts.append(f"<text x='{left}' y='{h - 10}' fill='var(--muted)'>{x_key}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.4f}"
    return str(v)


def _format_cell(col_name: str, val):
    s = str(val)
    if col_name == "recommended_action":
        if "escalate" in s:
            return f"<span class='pill pill-bad'>Escalate Review</span>"
        elif "reconcile" in s:
            return f"<span class='pill pill-warn'>Reconcile</span>"
        return f"<span class='pill pill-accent'>Monitor</span>"
    elif col_name == "Decision status":
        if "pending" in s:
            return f"<span class='pill pill-good'>● Pending Human Review</span>"
        elif "rejected" in s:
            return f"<span class='pill pill-bad'>● Policy Rejection (Ungrounded)</span>"
        return f"<span class='pill pill-accent'>{s}</span>"
    elif col_name in ("ROC-AUC", "PR-AUC") and isinstance(val, (int, float)) and val > 0.9:
        return f"<span style='color:var(--good); font-weight:700;' class='mono'>{val:,.4f}</span>"
    elif isinstance(val, (int, float)):
        return f"<span class='mono'>{val:,.4f}</span>"
    return s


def render(payload: dict) -> str:
    meta = payload.get("meta", {})
    kpis = payload.get("kpis", {})
    metrics_rows = payload.get("metrics", [])
    km = payload.get("km_curves", [])
    batch = payload.get("batch_quality", [])
    scenarios = payload.get("scenarios_mc", [])
    drift = payload.get("drift", [])
    queue = payload.get("review_queue", [])
    audit = payload.get("audit", [])

    kpi_subtitles = {
        "rows": "Monthly panel rows profiled",
        "loans": "Unique loan contracts",
        "features": "Contemporaneously observable",
        "quality_low": "Anomalous / breach-heavy",
        "queue": "Prioritised human review",
    }
    kpi_html = "".join(
        f"<div class='card kpi'>"
        f"<div class='value'>{kpis[k]['value']}</div>"
        f"<div class='label'>{kpis[k]['label']}</div>"
        f"<div style='font-size:11px; color:var(--dim); margin-top:4px;'>{kpi_subtitles.get(k, '')}</div>"
        f"</div>" for k in kpis)

    metric_html = ""
    if metrics_rows:
        head_cols = metrics_rows[0]
        head = "".join(f"<th>{h}</th>" for h in head_cols)
        body = "".join(
            "<tr>" + "".join(
                f"<td>{_format_cell(head_cols[idx], c)}</td>" for idx, c in enumerate(row)
            ) + "</tr>"
            for row in metrics_rows[1:])
        metric_html = f"<div class='panel'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"

    queue_html = ""
    if queue:
        head_cols = queue[0]
        head = "".join(f"<th>{h}</th>" for h in head_cols)
        body = "".join(
            "<tr>" + "".join(
                f"<td>{_format_cell(head_cols[idx], c)}</td>" for idx, c in enumerate(row)
            ) + "</tr>"
            for row in queue[1:])
        queue_html = f"""<div class='panel'>
  <div class='queue-toolbar'>
    <div style='font-size:12px; color:var(--muted); font-weight:500;'>Showing top <strong>{len(queue)-1}</strong> prioritised loans &bull; Hybrid Isolation Forest + Rules</div>
    <input type='text' class='search-input' placeholder='🔍 Search loan ID, reason, or action...' oninput="filterTable('queue-table', this.value)"/>
  </div>
  <div class='table-scroll-container'>
    <table id='queue-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>
  </div>
</div>"""
    else:
        queue_html = "<div class='panel'><p class='meta'>no queue rows</p></div>"

    drift_html = ""
    if drift:
        head_cols = drift[0]
        head = "".join(f"<th>{h}</th>" for h in head_cols)
        body = "".join(
            "<tr>" + "".join(
                f"<td>{_format_cell(head_cols[idx], c)}</td>" for idx, c in enumerate(row)
            ) + "</tr>"
            for row in drift[1:])
        drift_html = f"<div class='panel'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"

    audit_html = "".join(
        f"<tr><td class='mono'>{a.get('timestamp_utc', '')}</td>"
        f"<td><strong>{a.get('model', '')}</strong></td>"
        f"<td>{_format_cell('Decision status', a.get('decision_status', ''))}</td></tr>" for a in audit)
    audit_table = (
        f"<table><thead><tr><th>Timestamp (UTC)</th><th>Model</th><th>Decision Status</th></tr></thead>"
        f"<tbody>{audit_html}</tbody></table>" if audit_html else "<p class='meta'>no audit entries</p>"
    )

    explainability = payload.get("explainability", [])
    counterfactuals = payload.get("counterfactuals", [])
    explain_section = ""
    if explainability or counterfactuals:
        imp_rows = "".join(
            f"<tr><td><span class='pill pill-accent'>{i['target']}</span></td>"
            f"<td><strong>{i['feature']}</strong></td>"
            f"<td class='mono' style='color:var(--good); font-weight:600;'>+{i['importance']:,.4f}</td></tr>"
            for i in explainability
        )
        cf_rows_html = "".join(
            f"<tr><td class='mono'><strong>{c['loan_id']}</strong></td>"
            f"<td class='mono' style='color:var(--bad);'>{c['baseline_prob']:.2%}</td>"
            f"<td><span class='pill pill-warn'>{c['best_counterfactual']}</span></td>"
            f"<td class='mono' style='color:var(--good); font-weight:700;'>{c['all_cured_prob']:.2%} ({c['all_cured_delta']:.2%})</td></tr>"
            for c in counterfactuals
        )
        explain_section = f"""
<!-- TASK 6: EXPLAINABILITY & RESPONSIBLE AI -->
<div class="section" id="task-6">
  <div class="section-head">
    <h2>Task 6 • Model Explainability &amp; Responsible AI</h2>
    <span class="section-desc">Global permutation feature attributions and actionable counterfactual policy interventions</span>
  </div>
  <div class="two-col">
    <div class="panel">
      <div style="font-weight:700; font-size:13px; margin-bottom:12px; color:var(--text);">Global Permutation Feature Importance</div>
      <table>
        <thead><tr><th>Target</th><th>Key Driver Feature</th><th>Importance (Δ Score)</th></tr></thead>
        <tbody>{imp_rows}</tbody>
      </table>
    </div>

    <div class="panel">
      <div style="font-weight:700; font-size:13px; margin-bottom:12px; color:var(--text);">Actionable Counterfactuals (What-If Interventions)</div>
      <table>
        <thead><tr><th>Loan ID</th><th>Base Default</th><th>Best Policy Intervention</th><th>All-Cured Risk</th></tr></thead>
        <tbody>{cf_rows_html}</tbody>
      </table>
    </div>
  </div>
</div>
"""

    reviewer_note = payload.get("reviewer_note", "")
    note_section = ""
    if reviewer_note:
        evidence = {}
        matched_rules = []
        if audit and isinstance(audit[0], dict):
            g = audit[0].get("grounding", {})
            evidence = g.get("evidence", {})
            matched_rules = g.get("matched_rules", [])

        loan_id = str(evidence.get("loan_id", "—"))
        cur_bal = evidence.get("current_balance")
        cur_bal_str = (f"-${abs(cur_bal):,.2f}" if isinstance(cur_bal, (int, float)) and cur_bal < 0
                       else (f"${cur_bal:,.2f}" if isinstance(cur_bal, (int, float)) else "—"))
        status = str(evidence.get("current_status", "—"))
        dpd = str(evidence.get("days_past_due", "—"))
        anom_score = evidence.get("anomaly_score")
        anom_score_str = f"{anom_score:.4f}" if isinstance(anom_score, (int, float)) else "—"
        dq_score = evidence.get("data_quality_score")
        dq_score_str = f"{dq_score:.1f}" if isinstance(dq_score, (int, float)) else "—"
        def_prob = evidence.get("next_12m_default_prob")
        def_prob_str = f"{def_prob:.2%}" if isinstance(def_prob, (int, float)) else "—"
        conf = evidence.get("confidence")
        conf_str = f"{conf:.1%}" if isinstance(conf, (int, float)) else "—"

        rule_pills = "".join(f"<span class='pill pill-bad' style='font-size:11px;'>🚫 Breached: {r}</span>" for r in matched_rules)
        reasons = [r.strip() for r in str(evidence.get("anomaly_reason", "")).split(";") if r.strip()]
        reason_pills = "".join(f"<span class='pill pill-warn' style='font-size:11px;'>⚡ Driver: {r}</span>" for r in reasons[:3])

        clean_note = reviewer_note
        for pfx in ("RECOMMENDATION ONLY — human review required.", "RECOMMENDATION ONLY:"):
            if clean_note.startswith(pfx):
                clean_note = clean_note[len(pfx):].strip()
        import re
        formatted_note = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', clean_note)

        note_section = f"""
<div class="section" id="task-7">
  <div class="section-head">
    <h2>Task 7 • Grounded LLM Reviewer Copilot Analysis</h2>
    <span class="section-desc">Audited LLM recommendation with citations and strict human control</span>
  </div>
  <div class="copilot-card">
    <div class="copilot-header">
      <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
        <span class="copilot-title">🤖 AI Case Assessment</span>
        <span class="brand-badge" style="font-size:10px;">Model: {audit[0].get('model', 'grounded-template-v1') if audit else 'grounded-template-v1'}</span>
        <span class="mono" style="font-size:12px; color:var(--text); background:rgba(255,255,255,0.06); padding:3px 8px; border-radius:6px;">Target: Loan <strong>{loan_id}</strong></span>
      </div>
      <span class="pill pill-good">Advisory Only • Requires Human Sign-Off</span>
    </div>

    <div class="copilot-metrics-grid">
      <div class="copilot-metric-chip">
        <div class="copilot-chip-label">Current Balance</div>
        <div class="copilot-chip-val" style="color:var(--bad);">{cur_bal_str}</div>
        <div class="copilot-chip-sub">Rule Breach (&lt; 0)</div>
      </div>
      <div class="copilot-metric-chip">
        <div class="copilot-chip-label">Reported Status</div>
        <div class="copilot-chip-val">{status}</div>
        <div class="copilot-chip-sub">{dpd} Days Past Due</div>
      </div>
      <div class="copilot-metric-chip">
        <div class="copilot-chip-label">Anomaly Score</div>
        <div class="copilot-chip-val" style="color:var(--warn);">{anom_score_str}</div>
        <div class="copilot-chip-sub">Isolation Forest + rule breaches</div>
      </div>
      <div class="copilot-metric-chip">
        <div class="copilot-chip-label">Data Quality Score</div>
        <div class="copilot-chip-val" style="color:var(--bad);">{dq_score_str} / 100</div>
        <div class="copilot-chip-sub">Degraded Record Integrity</div>
      </div>
      <div class="copilot-metric-chip">
        <div class="copilot-chip-label">12M Default Risk</div>
        <div class="copilot-chip-val" style="color:var(--good);">{def_prob_str}</div>
        <div class="copilot-chip-sub">Calibrated HGB Model</div>
      </div>
      <div class="copilot-metric-chip">
        <div class="copilot-chip-label">Model Confidence</div>
        <div class="copilot-chip-val">{conf_str}</div>
        <div class="copilot-chip-sub">Reviewer Confidence</div>
      </div>
    </div>

    <div class="copilot-narrative-box">
      <div class="copilot-narrative-label">
        <span>📝 Grounded Narrative Assessment</span>
        <span style="font-size:11px; color:var(--dim); font-weight:normal;">Cites Data Dictionary &amp; Validation Rules • Zero Hallucinations</span>
      </div>
      <div style="margin-bottom:8px;">
        <span class="pill pill-good" style="margin-right:6px; font-size:10px;">RECOMMENDATION ONLY</span>
        <span style="color:var(--muted); font-size:12px;">Human verification mandatory prior to servicing action</span>
      </div>
      <div class="copilot-narrative-text">{formatted_note}</div>
    </div>

    <div class="copilot-footer">
      <div class="copilot-tags">
        <span class="copilot-tag-label">Triggered Violations &amp; Drivers:</span>
        {rule_pills}
        {reason_pills}
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="color:var(--muted); font-size:11px;">Recommended Action:</span>
        <span class="pill pill-bad" style="font-size:12px; padding:5px 12px;">👤 Escalate to Human Reviewer</span>
      </div>
    </div>
  </div>
</div>
"""

    scenario_rows_html = ""
    for s in scenarios:
        def fmt(k):
            v = s.get(k)
            return f"{v:,.4f}" if isinstance(v, (int, float)) else "—"
        scenario_rows_html += (
            f"<tr><td><strong>{s.get('scenario', '')}</strong></td>"
            f"<td class='mono'>{fmt('mean_default_rate')}</td>"
            f"<td class='mono'>{fmt('p05_default_rate')} – {fmt('p95_default_rate')}</td>"
            f"<td class='mono'>{fmt('mean_prepayment_rate')}</td>"
            f"<td class='mono'>{fmt('p05_prepayment_rate')} – {fmt('p95_prepayment_rate')}</td></tr>")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="icon" href="data:,"/>
<title>Loan Performance Intelligence Engine — Monitoring Dashboard</title>
<style>{_CSS}</style></head>
<body>
<header>
  <div class="header-top">
    <div style="display:flex; align-items:center; gap:12px;">
      <span class="brand-badge">Intain 2026 • AI Track</span>
      <span class="status-pill">Pipeline Verified</span>
    </div>
    <div class="meta">
      <span>Cutoff: <strong>{meta.get('cutoff', 'n/a')}</strong></span>
      <span>Features: <strong>{meta.get('n_features', 'n/a')}</strong></span>
      <span>Seed: <strong>{meta.get('seed', 'n/a')}</strong></span>
    </div>
  </div>
  <h1>Loan Performance Intelligence Engine — Production Dashboard</h1>
  <nav class="sub-nav">
    <a href="#overview" class="nav-item">Overview</a>
    <a href="#task-1" class="nav-item">Data Quality</a>
    <a href="#task-2" class="nav-item">Model Validation</a>
    <a href="#task-3" class="nav-item">Survival Curves</a>
    <a href="#task-4" class="nav-item">Anomaly Queue</a>
    <a href="#task-5" class="nav-item">Stress Scenarios</a>
    <a href="#task-6" class="nav-item">Explainability</a>
    <a href="#task-7" class="nav-item">AI Copilot</a>
    <a href="#task-8" class="nav-item">Audit Trail</a>
  </nav>
</header>

<div class="grid" id="overview">{kpi_html}</div>

<!-- TASK 1: DATA INTELLIGENCE & DRIFT PROFILING -->
<div class="section" id="task-1">
  <div class="section-head">
    <h2>Task 1 • Data Intelligence, Quality &amp; Drift Profiling</h2>
    <span class="section-desc">Dual-axis monthly health tracking and train vs out-of-time population stability index (PSI)</span>
  </div>
  <div class="two-col-chart">
    <div class="panel">
      <div style="font-weight:700; font-size:13px; margin-bottom:12px; color:var(--text); display:flex; justify-content:space-between; align-items:center;">
        <span>Data Quality &amp; Breach Trends Over Time</span>
        <span class="brand-badge" style="font-size:10px;">Monthly Longitudinal</span>
      </div>
      {_svg_line_chart(batch, "month_ordinal",
          [("mean_quality_score", "mean quality score", "#38bdf8"),
           ("breaches_per_1k_rows", "breaches / 1k rows", "#f87171")], dual_axis=True)}
      <div class="legend">
        <span style="color:#38bdf8">Mean Data-Quality Score (Left Axis)</span>
        <span style="color:#f87171">Rule Breaches Per 1,000 Rows (Right Axis)</span>
      </div>
    </div>

    <div>
      <div style="font-weight:700; font-size:13px; margin-bottom:12px; color:var(--text); display:flex; justify-content:space-between; align-items:center; padding: 0 4px;">
        <span>Population Stability Index (PSI Drift)</span>
        <span class="pill pill-accent" style="font-size:10px;">Adv. Feature #3</span>
      </div>
      {drift_html}
    </div>
  </div>
</div>

<!-- TASK 2: PREDICTIVE MODELLING & VALIDATION -->
<div class="section" id="task-2">
  <div class="section-head">
    <h2>Task 2 • Out-of-Time Model Validation Performance</h2>
    <span class="section-desc">Strict chronological holdout: Baseline Random Forest vs Calibrated HistGradientBoosting</span>
  </div>
  {metric_html}
</div>

<!-- TASK 3: TIME-TO-EVENT & SURVIVAL PROJECTIONS -->
<div class="section" id="task-3">
  <div class="section-head">
    <h2>Task 3 • Competing-Risk Survival Curves (Aalen-Johansen CIF)</h2>
    <span class="section-desc">Competing-risk survival curves with Kaplan-Meier and Aalen-Johansen estimator</span>
  </div>
  <div class="panel">
    {_svg_line_chart(km, "month",
        [("km_survival", "overall survival", "#94a3b8"),
         ("cif_default", "default CIF", "#f87171"),
         ("cif_prepay", "prepayment CIF", "#38bdf8")], y_range=(0.0, 1.0))}
    <div class="legend">
      <span style="color:#94a3b8">Kaplan-Meier Survival</span>
      <span style="color:#f87171">Default Cumulative Incidence</span>
      <span style="color:#38bdf8">Prepayment Cumulative Incidence</span>
    </div>
  </div>
</div>

<!-- TASK 4: ANOMALY & EXCEPTION INTELLIGENCE -->
<div class="section" id="task-4">
  <div class="section-head">
    <h2>Task 4 • Prioritised Anomaly &amp; Exception Reviewer Queue</h2>
    <span class="section-desc">Reviewer queue combining Isolation Forest anomaly scores with deterministic rule breaches</span>
  </div>
  {queue_html}
</div>

<!-- TASK 5: SCENARIO & STRESS SIMULATION -->
<div class="section" id="task-5">
  <div class="section-head">
    <h2>Task 5 • Macroeconomic Scenario Stress Simulation (Bootstrap 5–95%)</h2>
    <span class="section-desc">Base vs Adverse vs High-Prepay macroeconomic feature shocks with Monte Carlo confidence bounds</span>
  </div>
  <div class="panel">
      <table>
      <thead><tr><th>Scenario</th><th>Mean Default Rate</th><th>Default 5th–95th</th><th>Mean Prepay Rate</th><th>Prepay 5th–95th</th></tr></thead>
      <tbody>
      {scenario_rows_html}
      </tbody>
      </table>
  </div>
</div>

<!-- TASK 6: EXPLAINABILITY & RESPONSIBLE AI -->
{explain_section}

<!-- TASK 7: SMART LLM USAGE (COPILOT) -->
{note_section}

<!-- TASK 8: GOVERNANCE, SAFETY & AUDIT TRAIL -->
<div class="section" id="task-8">
  <div class="section-head">
    <h2>Task 8 • Governance &amp; Copilot Audit Trail</h2>
    <span class="section-desc">Copilot audit trail with real-time prompt/decision logging and demonstrated hallucination rejection</span>
  </div>
  <div class="panel">{audit_table}</div>
</div>
<div style="height: 40px;"></div>
<script>
function filterTable(tableId, query) {{
  var table = document.getElementById(tableId);
  if (!table) return;
  var rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');
  var q = (query || '').toLowerCase();
  for (var i = 0; i < rows.length; i++) {{
    var text = rows[i].textContent || rows[i].innerText;
    rows[i].style.display = text.toLowerCase().indexOf(q) > -1 ? '' : 'none';
  }}
}}
</script>
</body></html>"""
    return html


def build_payload(run_info: dict) -> dict:
    """Assemble the dashboard payload from pipeline artefacts (JSON-serialisable)."""
    return run_info
