## Description: <br>
Measure or improve the runtime, startup time, memory use, or throughput of a Newton simulation that already behaves correctly, or review a Newton step loop for performance problems such as CUDA graph capture mistakes and host synchronization. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers using released Newton physics simulator wheels use this skill to scope, profile, and optimize the performance of an already-correct Newton simulation, or to review a Newton step loop for CUDA graph capture and host synchronization problems, with recommendations routed to the version-matched Newton and Warp documentation. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Newton physics simulator (GitHub)](https://github.com/newton-physics/newton) <br>
- [Newton installation guide](https://newton-physics.github.io/newton/latest/guide/installation.html) <br>
- [Newton documentation (version-matched root: newton-physics.github.io/newton/<version>/)](https://newton-physics.github.io/newton/latest/) <br>
- [Warp documentation (version-matched root: nvidia.github.io/warp/v<major>.<minor>/)](https://nvidia.github.io/warp/stable/) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Shell commands] <br>
**Output Format:** [Markdown with inline Python and shell code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Reports Newton and Warp versions, documentation cited, timing scope, before/after measurements with variance, and fidelity evidence; proposed experiments are labeled separately from collected measurements.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
12 evaluation tasks (11 positive, 1 negative), 1 attempt per task, run in the `k8s-sandbox` environment with each task attempt in its own isolated sandbox pod; each task was also attempted without the skill as a baseline. Dataset digest: `sha256:6ecce6eb8f7b83630e185404459abc12d2a3d0ce0ab5c386ad9dbe1f77ef09f9` (skill-evaluator-dataset-snapshot/1). Evaluator version `1.6.9`; evaluation date 2026-11-11. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 87.3% — baseline ran, but no comparable score was available; uplift unavailable | 84.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 88.3% → 91.7% (+3.4 points) | 93.3% → 98.3% (+5.0 points) |
| Discoverability | 65.5% — baseline ran, but no comparable score was available; uplift unavailable | 74.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 87.5% → 83.8% (-3.7 points) | 94.2% → 95.8% (+1.6 points) |
| Efficiency | 95.5% — baseline ran, but no comparable score was available; uplift unavailable | 54.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.0% → 97.6% (-0.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.1% → 85.5% (-2.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Skill Version(s): <br>
de81080 (source: git SHA, committed 2026-11-04) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
