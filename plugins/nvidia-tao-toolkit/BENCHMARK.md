# nvidia-tao-toolkit · Benchmark

> ✅ **PASS**  
> Passes all four evaluation score checks; Claude Code's partial Effectiveness assessment and Codex's Efficiency regression need attention.  
> Tested date not recorded · v0.1.19 · Claude Code, Codex · 139 tasks × 1 attempt

## Does it help?

Yes. Average lift is +70.56 points on Claude Code (largest baseline improvement) and +17.73 on Codex. Codex has the highest with-plugin quality score: 98.06/100.

| Dimension | Claude Code: score with plugin (lift) | Codex: score with plugin (lift) |
| --- | --- | --- |
| Security | ✅ 100.00 (+100.00) | ✅ 100.00 (+0.00) |
| Correctness | ✅ 91.85 (+58.09) | ✅ 97.60 (+57.07) |
| Effectiveness | ✅ 95.93 (lift unavailable; partial coverage) | ✅ 97.71 (baseline not applicable) |
| Efficiency | ✅ 90.99 (tokens: lift unavailable) | ✅ 95.72 (tokens: lift unavailable) |
| Discoverability | ✅ 94.78 (no baseline) | ✅ 99.28 (no baseline) |
| **Overall score** | **94.71** | **98.06** |
| **Average lift** | **+70.56** | **+17.73** |

Score: with the bundled skills. Lift: change from the same tasks run without them. Efficiency lift compares token use only; Discoverability has no baseline. ✅ 50 or above · ⚠️ 40 to 49 · ❌ below 40.

## Publication checks

| Check | Result |
| --- | --- |
| No blocking evaluated security issues; Security ≥ 50 on every agent | ✅ Recorded Security PASS; 100.00 on both agents. |
| At least one agent passes every dimension (≥ 50) | ✅ Codex, with complete score coverage. |
| Better than without the plugin (largest average lift ≥ +5) | ✅ +70.56 (Claude Code). |
| Highest with-plugin quality score ≥ 80 | ✅ 98.06 (Codex). |

## Watch out (2)

- ⚠️ **Codex's Efficiency score decreases with the plugin.** 99.61 without it versus 95.72 with it (−3.89 points). → Review the paired task evidence for repeated or unnecessary work.
- ⚠️ **Claude Code's Effectiveness is partially assessed.** Its 95.93 score uses available observations; Codex's score coverage is complete. → Review the retained Effectiveness assessments when interpreting Claude Code's score.

## Known limits

- One attempt per task; only Claude Code and Codex tested.
- Evaluation covers bundled skills. Native plugin loading, hooks, root instructions and display-name behavior were not evaluated.
- Effectiveness baseline lift is unavailable for Claude Code and not applicable for Codex; token-only Efficiency lift is unavailable.

<details>
<summary>How this was scored</summary>

139 tasks, two agents, baseline and with-plugin variants: 556 results. All 556 executions completed; 555 were fully graded. The combined export retains 528 earlier results and 28 recovery results, with no pending task slots.

Scores are out of 100 and are copied from the saved canonical assessment. Under its recorded policy, Security uses the lowest usable task score; other dimensions use means of usable applicable observations. Each agent's overall score equally weights its available dimensions. Claude Code's score coverage is partial; Codex's is complete. The saved aggregate is 94.71/100, PASS with partial score coverage, and remains the minimum across scored agents.

Average lift follows the saved producer's comparable Security, Correctness and Efficiency score dimensions. Efficiency score lift is +53.60 points on Claude Code and −3.89 on Codex; those are score changes, not token-use lift. Claude Code's paired comparisons use 138 of 139 task pairs; Codex's use 139 of 139. Full current scores and paired comparisons can therefore use different populations.

</details>

<details>
<summary>Provenance</summary>

| Item | Value |
| --- | --- |
| Dataset digest (SHA-256) | Not recorded in shared-result dataset entries. |
| Evaluator version | NeMo Gym 0.7.0 |
| Evaluated agents | Claude Code and Codex |
| Plugin version | 0.1.19 |
| Evaluated source revision | Not recorded in the evaluation export. |
| Quality / lift policy | `nvcarps-five-dimensions-v6` / `nvcarps-best-agent-baseline-lift-v2` |
| Evaluation result (SHA-256) | `04ee285e7025dbfdec33499941c39c47ca3589490d8f27bda2eec1affa08802f` |

</details>
