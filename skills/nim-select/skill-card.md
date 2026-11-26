## Description: <br>
Checks if an NVIDIA NIM fits a GPU host or cluster before a pull, never from memory; lists models that fit; picks the NGC or China distributor image, tag and profile. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to check, from script output rather than memory, whether an NVIDIA NIM fits their GPU host or Kubernetes cluster, to list the NIMs that fit, and to pick the pinned NGC or China NIM distributor image, tag and profile before handing off to deployment. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [China NIM distributors](references/china-distributors.md) <br>
- [Fit rules](references/fit-rules.md) <br>
- [List mode](references/list-mode.md) <br>
- [NIM Metadata API](references/metadata-api.md) <br>
- [Security](references/security.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Files] <br>
**Output Format:** [Markdown with inline bash code blocks; scripts emit JSON facts files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Read-only: never pulls, runs, deploys or logs in to a registry; reports script verdicts, images and pinned version tags as given] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
36 evaluation tasks (36 positive), 1 attempt per task, each in its own isolated sandbox pod in the `k8s-sandbox` environment (dataset digest `sha256:56820f16a7580a22f2ebc2adb6773432ffded122b6fcec60da496513b75dd797`, evaluator version `1.6.10`, evaluated 2026-11-26). Baseline is the same task attempted without the target skill. <br>

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
| Overall | 87.9% — baseline ran, but no comparable score was available; uplift unavailable | 85.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 93.1% → 98.6% (+5.5 points) | 83.3% → 95.8% (+12.5 points) |
| Correctness | 34.4% → 90.6% (+56.2 points) | 28.9% → 91.7% (+62.8 points) |
| Discoverability | 89.8% — baseline ran, but no comparable score was available; uplift unavailable | 89.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 19.8% → 65.9% (+46.1 points) | 19.2% → 59.9% (+40.7 points) |
| Efficiency | 94.5% — baseline ran, but no comparable score was available; uplift unavailable | 89.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.8% → 91.3% (-6.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.1% → 89.1% (+5.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 70 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 36 task(s) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
