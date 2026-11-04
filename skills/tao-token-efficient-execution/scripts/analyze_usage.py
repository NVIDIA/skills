#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenomics accounting for Pi agent session JSONL files.

Implements docs/MEASURE_FIRST.md from the tokenomics repo against Pi's session
format (~/.pi/agent/sessions/... or the kit's --session-dir):

  entry:   {"type": "message", "message": {"role": "assistant", "usage": {...}}}
  usage:   {"input", "output", "cacheRead", "cacheWrite", "totalTokens",
            "cost": {"input", "output", "cacheRead", "cacheWrite", "total"}}

Metrics (input-token equivalents):
  bill          = output*5 + cacheRead*0.1 + cacheWrite*1.25 + input*1
  no-cache bill = same with cacheRead at *1
  token-miles   = sum(cacheRead)
  peak context  = max(input + cacheRead + cacheWrite) over messages

Usage: python3 analyze_usage.py <session-dir-or-file> [more ...] [--per-message]
"""
import json
import sys
from pathlib import Path

W_OUT, W_READ, W_WRITE, W_IN = 5.0, 0.1, 1.25, 1.0


def iter_files(args):
    for a in args:
        p = Path(a).expanduser()
        if p.is_dir():
            yield from sorted(p.glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
        elif p.is_file():
            yield p
        else:
            print(f"warning: {a} not found", file=sys.stderr)


def session_stats(path, per_message=False):
    s = {
        "file": path.name, "messages": 0, "input": 0, "output": 0,
        "cache_read": 0, "cache_write": 0, "cache_write_1h": 0,
        "peak_ctx": 0, "native_cost": 0.0,
    }
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = entry.get("message") if entry.get("type") == "message" else None
            usages = []
            if isinstance(msg, dict):
                u = msg.get("usage")
                if isinstance(u, dict) and msg.get("role") == "assistant":
                    usages.append(u)
                # nested LLM work performed by tools (subagents etc.)
                if isinstance(u, dict) and msg.get("role") == "toolResult":
                    usages.append(u)
            # compaction / branch-summary entries carry generation usage too
            elif entry.get("type") in ("compaction", "branch_summary") and isinstance(entry.get("usage"), dict):
                usages.append(entry["usage"])
            for u in usages:
                inp = int(u.get("input") or 0)
                out = int(u.get("output") or 0)
                cr = int(u.get("cacheRead") or 0)
                # 1h-TTL cache writes (cacheWrite1h) are priced at x2, not x1.25;
                # fold them into cache_write volume but bill them separately.
                cw = int(u.get("cacheWrite") or 0)
                cw1h = int(u.get("cacheWrite1h") or 0)
                s["messages"] += 1
                s["input"] += inp
                s["output"] += out
                s["cache_read"] += cr
                s["cache_write"] += cw
                s["cache_write_1h"] += cw1h
                s["peak_ctx"] = max(s["peak_ctx"], inp + cr + cw + cw1h)
                cost = u.get("cost") or {}
                s["native_cost"] += float(cost.get("total") or 0.0)
                if per_message:
                    print(f"    msg in={inp} out={out} cr={cr} cw={cw} ctx={inp+cr+cw}")
    return s


W_WRITE_1H = 2.0


def bill(s, cache=True):
    read_w = W_READ if cache else 1.0
    return (s["output"] * W_OUT + s["cache_read"] * read_w + s["cache_write"] * W_WRITE
            + s.get("cache_write_1h", 0) * W_WRITE_1H + s["input"] * W_IN)


def fmt(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(int(n))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    per_message = "--per-message" in sys.argv
    if not args:
        print(__doc__)
        sys.exit(2)

    files = list(iter_files(args))
    if not files:
        print("no session files found", file=sys.stderr)
        sys.exit(1)

    total = {"messages": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "cache_write_1h": 0, "peak_ctx": 0, "native_cost": 0.0}
    n_sessions = 0
    print(f"{'session':<42} {'msgs':>5} {'output':>8} {'uncached':>9} {'cacheR':>8} {'cacheW':>8} {'peak_ctx':>9} {'bill':>9} {'no-cache':>9} {'cost$':>8}")
    for f in files:
        s = session_stats(f, per_message=per_message)
        if s["messages"] == 0:
            continue
        n_sessions += 1
        for k in ("messages", "input", "output", "cache_read", "cache_write", "cache_write_1h", "native_cost"):
            total[k] += s[k]
        total["peak_ctx"] = max(total["peak_ctx"], s["peak_ctx"])
        name = s["file"][:40]
        print(f"{name:<42} {s['messages']:>5} {fmt(s['output']):>8} {fmt(s['input']):>9} {fmt(s['cache_read']):>8} "
              f"{fmt(s['cache_write'] + s['cache_write_1h']):>8} {fmt(s['peak_ctx']):>9} {fmt(bill(s)):>9} {fmt(bill(s, cache=False)):>9} {s['native_cost']:>8.3f}")
    carrying = total["cache_read"] * W_READ + total["cache_write"] * W_WRITE + total["cache_write_1h"] * W_WRITE_1H
    b = bill(total)
    print("-" * 132)
    print(f"{'TOTAL (' + str(n_sessions) + ' sessions)':<42} {total['messages']:>5} {fmt(total['output']):>8} {fmt(total['input']):>9} "
          f"{fmt(total['cache_read']):>8} {fmt(total['cache_write'] + total['cache_write_1h']):>8} {fmt(total['peak_ctx']):>9} {fmt(b):>9} {fmt(bill(total, cache=False)):>9} {total['native_cost']:>8.3f}")
    print()
    print(f"bill (input-eq):      {fmt(b)}")
    print(f"no-cache bill:        {fmt(bill(total, cache=False))}")
    print(f"token-miles:          {fmt(total['cache_read'])}")
    print(f"peak context:         {fmt(total['peak_ctx'])}")
    print(f"carrying share:       {100.0 * carrying / b if b else 0:.1f}%")
    print(f"native cost (USD):    ${total['native_cost']:.3f}")


if __name__ == "__main__":
    main()
