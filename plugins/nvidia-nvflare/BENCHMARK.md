# nvidia-nvflare · Benchmark

> ✅ **PASS**
>
> Evaluation passes with partial coverage; static findings and Efficiency regressions need attention.
>
> v1.0.0 · Claude Code, Codex · 5 task cases

## Does it help?

Yes. Codex has the highest with-plugin quality score, **80.66/100**,
and the largest average lift, **+37.27 points**. Claude Code's average lift is
**+6.63 points**. Both agents have partial score coverage.

| Dimension | Claude Code: score with plugin (lift) | Codex: score with plugin (lift) |
| --- | --- | --- |
| Security | ✅ 100.00 (+0.00) | ✅ 100.00 (+100.00) |
| Correctness | ✅ 73.33 (+30.00) | ✅ 70.83 (+22.22) |
| Effectiveness | ✅ 55.56 (lift unavailable) | ✅ 58.33 (lift unavailable) |
| Efficiency | ✅ 80.80 (-10.11) | ✅ 74.13 (-10.42) |
| Discoverability | ✅ 80.00 (no baseline) | ✅ 100.00 (no baseline) |
| **Overall score** | **77.94** | **80.66** |
| **Average lift** | **+6.63** | **+37.27** |

Scores are out of 100 with the bundled skills. Lift compares matched tasks
without and with the plugin; its paired population can differ from the full
score population. Efficiency lift above is a score change, not a token-use
change. ✅ 50 or above · ⚠️ 40 to below 50 · ❌ below 40.

## Publication checks

Evaluation PASS describes scored behavior. Static validation remains a separate
check; this table does not grant publication approval.

| Check | Result |
| --- | --- |
| Recorded evaluation Security ≥ 50 on every agent | ✅ 100.00 on both agents; partial coverage. |
| At least one agent passes every dimension (≥ 50) | ✅ Codex. |
| Better than without the plugin (largest average lift ≥ +5) | ✅ +37.27 (Codex). |
| Highest with-plugin quality score ≥ 80 | ✅ 80.66 (Codex). |
| Static validation | ⚠️ **Incomplete coverage** |

## Watch out (3)

- ⚠️ **Static coverage needs attention.** Archive inspection, encoded content and connection analysis have incomplete coverage. The full validation report retains the findings and detailed diagnostics.
- ⚠️ **Grading and execution are partial.** Execution completed for 17 of 20 task slots; 12 of 20 were fully graded.
- ⚠️ **Efficiency scores decrease on matched tasks.** Claude Code: −10.11 points; Codex: −10.42 points. Review the paired results when comparing performance.

## Known limits

- Five task cases; only Claude Code and Codex tested.
- Evaluation covers bundled-skill tasks. Native plugin loading and full coverage of all nine declared skills were not established.
- Both agents have partial Security coverage. Codex also has partial Correctness, Effectiveness, Efficiency and Discoverability coverage; Claude Code's Effectiveness coverage is partial.
- Effectiveness has no usable matched baseline; Discoverability has no baseline. Efficiency lift reports score points, not token-use changes.

<details>
<summary>How this was scored</summary>

Five cases, two agents, baseline and with-plugin variants: 20 task slots.
Incomplete and unsuccessful attempts remain in the retained evidence; missing
observations are not scored as zero.

Scores and lift follow the retained assessment. Security uses the lowest usable
task score; the other
dimensions use means of usable applicable observations. Each agent's overall
score equally weights its available dimensions. Codex meets the saved agent
score requirement of 80; Claude Code scores 77.94. The saved aggregate remains
**77.94/100, PASS with partial coverage**, using the minimum across scored agents.
It is distinct from Codex's individual score of 80.66.

Average lift uses the saved comparable Security, Correctness and Efficiency
dimensions. Included / expected task pairs are:

| Dimension | Claude Code | Codex |
| --- | --- | --- |
| Security | 3/5 | 2/5 |
| Correctness | 5/5 | 3/5 |
| Efficiency | 5/5 | 3/5 |

Current-quality scores and paired lift can therefore use different populations.
Execution and grading completion are separate from the quality verdict.

</details>

<details>
<summary>Provenance</summary>

| Item | Value |
| --- | --- |
| Evaluator version | NeMo Gym 0.7.0 |
| Evaluated agents | Claude Code and Codex |
| Plugin version | 1.0.0 |
| Validation tools | CLI 1.3.1435; Plugin Verifier 0.1.10 |

</details>
