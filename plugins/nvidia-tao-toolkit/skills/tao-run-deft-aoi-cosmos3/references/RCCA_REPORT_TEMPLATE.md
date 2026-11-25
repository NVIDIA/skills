# Cosmos3 Proxy RCCA Report: `<baseline|iterN>`

Build this report from `gaps_summary.json`, `false_accepts.json`, and
`false_rejects.json`. Replace the guidance below with concrete evidence before
committing `proxy_rcca`. The machine-readable artifact and heading contract is
`references/rcca-artifact-manifest.json`.

## 1. Verdict

State KPI reachability, Proxy accuracy/RCCA numbers, and the headline finding.

## 2. False-Accept Breakdown

Summarize counts and share by defect type from `false_accepts.json`.

| Defect type | Count | Share |
|---|---:|---:|
| `<type>` | `<count>` | `<percent>` |

## 3. False-Reject Breakdown

Summarize counts and share by defect type from `false_rejects.json`.

| Defect type | Count | Share |
|---|---:|---:|
| `<type>` | `<count>` | `<percent>` |

## 4. Top-K Worst Samples

List the worst sample IDs and explain why each is high priority.

## 5. Per-Defect Analysis

| Defect type | False accepts | False rejects | RCCA finding |
|---|---:|---:|---|
| `<type>` | `<count>` | `<count>` | `<finding>` |

## 6. Recommended Actions

Name concrete mining targets and SDG defect/count targets for the next
iteration.
