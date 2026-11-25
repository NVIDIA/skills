# nvidia-doca-plugin · Benchmark

> ✅ **PASS**
> Passes all four publication checks; token use, skill selection and partial assessments need attention.
> Tested date not recorded · v1.0.0 · Claude Code, Codex · 24 tasks × 1 attempt · Self-assessed

## Does it help?

Yes. Average lift is +69.4 points on Codex (best agent) and +68.8 on Claude Code.

| Dimension | Claude Code: score with skill (lift) | Codex: score with skill (lift) |
|---|---|---|
| Security | ✅ 100.0 (+100.0) | ✅ 100.0 (+100.0) |
| Correctness | ✅ 74.3 (+37.5) | ✅ 93.8 (+38.9) |
| Effectiveness | ✅ 73.5 (no baseline) | ✅ 95.5 (no baseline) |
| Efficiency | ✅ 98.9 (tokens: lift unavailable) | ✅ 99.2 (tokens: lift unavailable) |
| Discoverability | ✅ 58.3 (no baseline) | ✅ 95.8 (no baseline) |
| **Overall score** | **81.0** | **96.9** |
| **Average lift** | **+68.8** | **+69.4** |

Score: with the skill installed. Lift: change from the same tasks run without it. Efficiency lift compares token use only; Discoverability has no baseline. ✅ 50 or above · ⚠️ 40 to 49 · ❌ below 40.

## Publication checks

| Check | Result |
|---|---|
| No blocking security issues; Security ≥ 50 on every agent | ✅ 0 blocking findings; Security 100.0 on both agents. |
| At least one agent passes every dimension (≥ 50) | ✅ Codex. |
| Better than without the skill (best agent ≥ +5; ⚠️ 0 to +5; ❌ below 0) | ✅ +69.4 (Codex). |
| Best agent score ≥ 80 | ✅ 96.9 (Codex). |

## Watch out (3)

- ⚠️ **Token use more than doubles.** +127.5% on Claude Code and +116.6% on Codex. → Review reference content that could load on demand.
- ⚠️ **Claude Code's Discoverability is near the bar.** 58.3 versus 50, with 10 of 24 tasks missing the expected skill. → Review skill descriptions against the missed prompts.
- ⚠️ **Claude Code's Effectiveness is partially assessed.** Four of 17 scored cases are incomplete. → Complete those assessments before treating the score as final.

## Known limits

- One attempt per task; only Claude Code and Codex tested.
- Activation or consultation recorded for 20 of 53 skills.
- No Effectiveness or Discoverability baseline; token-only Efficiency lift unavailable.

<details>
<summary>How this was scored</summary>

24 tasks, two agents, baseline and with-plugin variants: 96 results. Scores are out of 100. Security uses the lowest task score; the other dimensions use task means. Overall score is the mean of the five dimension scores. Baseline Security is 0 on both agents under the minimum rule, producing the +100-point lift. Token changes compare total input plus output tokens without adding cached tokens again.

</details>

<details>
<summary>Provenance: dataset digest, evaluator version, environment</summary>

| Item | Value |
|---|---|
| Dataset digest (SHA-256) | `d868e3253cce75a6b0b5106c92c5c90b978bc078921518b12ee35119287ef7cf` |
| Evaluator version | NeMo Gym 0.7.0 |
| Environment | Submitted local execution (Self-assessed) |

</details>
