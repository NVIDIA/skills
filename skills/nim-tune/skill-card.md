## Description: <br>
Tunes a running vLLM-backed NVIDIA NIM's runtime settings on measured AIPerf evidence, one setting per candidate, against the untuned NIM, on the NIM Operator or Docker. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to tune or speed up a running vLLM-backed NIM's engine settings or speculative decoding for a given GPU budget, workload and latency target, and to recommend the fastest measured configuration that still meets that target. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Other [Kubernetes cluster access]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Campaign file](references/campaign.md) <br>
- [Guard hook](references/hook.md) <br>
- [NIM runtime knobs](references/nim-knobs.md) <br>
- [Planning tuning candidates](references/planning.md) <br>
- [recommended-config.schema.json](references/recommended-config.schema.json) <br>
- [NIM for LLMs environment variables](https://docs.nvidia.com/nim/large-language-models/latest/reference/environment-variables.html) <br>
- [NIM for LLMs advanced configuration](https://docs.nvidia.com/nim/large-language-models/latest/reference/advanced-configuration.html) <br>
- [NIM for LLMs release notes](https://docs.nvidia.com/nim/large-language-models/latest/about-nim-llm/release-notes.html) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks; JSON campaign and recommended-config files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Scripts print one JSON object; credentials are referenced, never written, and reports exclude raw prompts and responses] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
24 evaluation tasks (24 positive), 1 attempt per task, each in its own isolated sandbox pod in the `k8s-sandbox` environment (evaluator version `1.6.10`, evaluated 2026-11-25). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 81.7% — baseline ran, but no comparable score was available; uplift unavailable | 81.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 91.7% → 91.7% (±0.0 points) | 83.3% → 95.8% (+12.5 points) |
| Correctness | 42.5% → 89.2% (+46.7 points) | 20.0% → 80.8% (+60.8 points) |
| Discoverability | 81.5% — baseline ran, but no comparable score was available; uplift unavailable | 90.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 36.5% → 53.3% (+16.8 points) | 22.7% → 47.2% (+24.5 points) |
| Efficiency | 92.7% — baseline ran, but no comparable score was available; uplift unavailable | 91.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.1% → 94.4% (-3.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.7% → 90.8% (+0.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
