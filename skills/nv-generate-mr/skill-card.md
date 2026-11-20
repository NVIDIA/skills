## Description: <br>
Used for generating synthetic body MRI volumes with NV-Generate-CTMR rflow-mr. Not for paired masks or production training data. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run the NV-Generate-CTMR rflow-mr workflow through a thin wrapper that stages config overrides, invokes the upstream inference entrypoint, and summarizes generated synthetic body MRI (T1, T2, FLAIR) NIfTI volumes for engineering verification. It is not for clinical deployment, clinical interpretation, autonomous diagnosis, regulatory submission, or production training data; upstream lists the NV-Generate-MR weights as NVIDIA Non-Commercial. <br>

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
- [FOV And Downloads](references/fov-and-downloads.md) <br>
- [NVIDIA-Medtech/NV-Generate-CTMR (pinned commit 61c4ec7)](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/tree/61c4ec709b84cad468852243c48e250bec732074) <br>
- [nvidia/NV-Generate-MR model weights (Hugging Face)](https://huggingface.co/nvidia/NV-Generate-MR) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [Synthetic MR NIfTI volumes under the caller-provided output directory plus a result JSON validated against validators/output_schema.json] <br>
**Output Parameters:** [3D (synthetic MR image volumes); 1D (result JSON)] <br>
**Other Properties Related to Output:** [Result JSON records the staged config, model inventory, upstream command, output geometry, spacing, affine, intensity range, and finite/non-constant data checks; outputs are synthetic and not clinically meaningful.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
2 evaluation tasks (1 positive, 1 negative), 1 attempt per task, in the `k8s-sandbox` environment with each task attempt in its own isolated sandbox pod. Tier 2 and Tier 3 evidence were required for publication. <br>

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
| Overall | Not available | 88.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 25.0% → 100.0% (+75.0 points) |
| Correctness | Not available | 0.0% → 100.0% (+100.0 points) |
| Discoverability | Not available | 80.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 16.7% → 66.7% (+50.0 points) |
| Efficiency | Not available | 94.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.4% → 89.6% (+0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 2 task(s) |

## Skill Version(s): <br>
0.1.0 (source: skill_manifest.yaml; git SHA 6c6743c, committed 2026-11-20) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
