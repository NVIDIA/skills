#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a run's status artifact: paper delta on top, best-result-so-far below.

The artifact answers two questions and nothing else:
  1. Does our pipeline reproduce the paper?  (delta table; n/p where nothing was published)
  2. What is the best number so far, and against what baseline?  (cards + bar list)

Deliberately stdlib-only so it runs anywhere the other scripts do.

    python render_status.py --status status.json --out status.html

Schema: see status.example.json. Every field is optional except `dataset`, `cards`
and `bars`; omitted sections are dropped rather than rendered empty.
"""
import argparse, html, json, sys
from pathlib import Path

CSS = """
:root{--surface-1:#faf9f7;--border:#e3e1db;--text-primary:#1a1a18;--text-secondary:#73726c;
--text-muted:#9b9a94;--bg-success:#d9edc7;--text-success:#2c5410;--bg-warning:#fadfaf;
--text-warning:#6b4108;--bg-accent:#cfe3f8;--text-accent:#12508f;--bar-muted:#8e8d86;
--bar-primary:#2c6b0e;--bar-secondary:#7a4a07;--radius:10px}
@media (prefers-color-scheme:dark){:root{--surface-1:#232320;--border:#3a3a36;
--text-primary:#f2f1ec;--text-secondary:#a8a79f;--text-muted:#7d7c75;--bg-success:#27500a;
--text-success:#c0dd97;--bg-warning:#633806;--text-warning:#fac775;--bg-accent:#0c447c;
--text-accent:#b5d4f4;--bar-muted:#6d6c66;--bar-primary:#97c459;--bar-secondary:#ef9f27}}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:16px;
line-height:1.7;color:var(--text-primary);max-width:900px;margin:0 auto;padding:24px;
background:transparent}
h2{font-size:18px;font-weight:500;margin:0 0 12px}
.card{border:0.5px solid var(--border);border-radius:var(--radius);padding:18px;margin-bottom:16px}
table{width:100%;border-collapse:collapse;font-size:15px}
th{color:var(--text-secondary);font-size:13px;font-weight:400;text-align:right;padding:4px 0}
th.l,td.l{text-align:left}
td{padding:8px 0;text-align:right;border-top:0.5px solid var(--border)}
.pill{padding:2px 8px;border-radius:6px;font-size:14px;background:var(--bg-success);
color:var(--text-success)}
.pill.warn{background:var(--bg-warning);color:var(--text-warning)}
.np{color:var(--text-muted)}
.note{font-size:13px;color:var(--text-muted);margin:12px 0 0}
.callout{background:var(--bg-accent);color:var(--text-accent);padding:14px 16px;
border-radius:var(--radius);font-size:15px;margin-bottom:30px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}
.big{border-radius:12px;padding:18px}
.big .lab{font-size:14px;margin-bottom:4px}
.big .val{font-size:44px;font-weight:500;line-height:1.1}
.big .sub{font-size:14px;margin:4px 0 12px}
.track{height:8px;background:rgba(255,255,255,.55);border-radius:4px;overflow:hidden}
.fill{height:100%}
.big .foot{font-size:13px;margin-top:8px}
.row{display:flex;align-items:center;gap:12px;margin-bottom:10px;font-size:15px}
.row .name{width:210px}
.row .track2{flex:1;height:22px}
.row .b{height:100%;border-radius:3px}
.row .v{width:64px;text-align:right}
.foot-note{font-size:13px;color:var(--text-muted);margin:0 0 24px}
"""

TONE = {"success": ("var(--bg-success)", "var(--text-success)"),
        "warning": ("var(--bg-warning)", "var(--text-warning)"),
        "accent": ("var(--bg-accent)", "var(--text-accent)")}
BAR = {"primary": "var(--bar-primary)", "secondary": "var(--bar-secondary)",
       "muted": "var(--bar-muted)"}


def e(x):
    return html.escape(str(x))


def gate_section(g, model, benchmark):
    if not g:
        return ""
    rows = []
    for r in g.get("rows", []):
        paper, delta = r.get("paper"), r.get("delta")
        pcell = f'<td class="np">n/p</td>' if paper in (None, "") else f"<td>{e(paper)}</td>"
        if delta in (None, ""):
            dcell = '<td class="np">—</td>'
        else:
            cls = "pill warn" if r.get("status") == "warn" else "pill"
            dcell = f'<td><span class="{cls}">{e(delta)}</span></td>'
        rows.append(f'<tr><td class="l">{e(r["metric"])}</td>'
                    f'<td>{e(r.get("measured", ""))}</td>{pcell}{dcell}</tr>')
    note = f'<p class="note">{e(g["note"])}</p>' if g.get("note") else ""
    title = f'1 · Delta vs paper — {e(model)} on {e(benchmark)}'
    return (f'<h2>{title}</h2><div class="card"><table>'
            f'<tr><th class="l">Metric</th><th>Measured</th><th>Paper</th><th>Delta</th></tr>'
            f'{"".join(rows)}</table>{note}</div>')


def big_cards(cards):
    out = []
    for c in cards:
        bg, fg = TONE.get(c.get("tone", "success"), TONE["success"])
        pct = max(0, min(1, float(c.get("progress", 0)))) * 100
        out.append(
            f'<div class="big" style="background:{bg};color:{fg}">'
            f'<div class="lab">{e(c.get("label",""))}</div>'
            f'<div class="val">{e(c["value"])}</div>'
            f'<div class="sub">{e(c.get("sub",""))}</div>'
            f'<div class="track"><div class="fill" style="width:{pct:.0f}%;background:{fg}"></div></div>'
            f'<div class="foot">{e(c.get("foot",""))}</div></div>')
    return f'<div class="cards">{"".join(out)}</div>'


def bar_list(bars):
    top = max([abs(float(b["value"])) for b in bars] + [1e-9])
    out = []
    for b in bars:
        w = abs(float(b["value"])) / top * 100
        color = BAR.get(b.get("tone", "muted"), BAR["muted"])
        strong = ' style="font-weight:500"' if b.get("tone") in ("primary", "secondary") else ""
        out.append(f'<div class="row"><div class="name"{strong}>{e(b["label"])}</div>'
                   f'<div class="track2"><div class="b" style="width:{w:.1f}%;background:{color}"></div></div>'
                   f'<div class="v"{strong}>{e(b["value"])}</div></div>')
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", required=True)
    ap.add_argument("--out", default="status.html")
    a = ap.parse_args()
    s = json.loads(Path(a.status).read_text())

    parts = [gate_section(s.get("gate"), s.get("model", "model"),
                          s.get("benchmark", "the paper's benchmark"))]
    if s.get("callout"):
        parts.append(f'<div class="callout">{e(s["callout"])}</div>')
    parts.append(f'<h2>2 · {e(s.get("result_title", "Best result so far"))} — '
                 f'{e(s.get("metric_name", "mAP@50-95"))} on {e(s["dataset"])}</h2>')
    parts.append(big_cards(s["cards"]))
    parts.append(bar_list(s["bars"]))
    if s.get("footnote"):
        parts.append(f'<p class="foot-note">{e(s["footnote"])}</p>')

    doc = f"<!doctype html><meta charset=utf-8><title>{e(s['dataset'])} status</title>" \
          f"<style>{CSS}</style><body>{''.join(parts)}</body>"
    Path(a.out).write_text(doc)
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
