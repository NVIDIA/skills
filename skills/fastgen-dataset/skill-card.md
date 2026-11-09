## Description: <br>
Prepare, register or repair datasets for NVIDIA FastGen loaders: class-conditional ZIPs, WebDataset shards of images, prompts, latents, videos, audio or camera data, and KD pairs or trajectories; then verify decoded batches. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers working with NVIDIA FastGen use this skill to prepare, register, or repair training and evaluation datasets (class-conditional ZIPs, WebDataset shards, cached latents, and knowledge-distillation pairs or trajectories) and to verify the decoded batches produced by the real loaders. <br>

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
- [Loader acceptance and stream contracts](references/loader-contract.md) <br>
- [FastGen dataset README](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/README.md) <br>
- [wds_dataloaders.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/wds_dataloaders.py) <br>
- [wds_utils.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/wds_utils.py) <br>
- [FastGen data configs (data.py)](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/data.py) <br>
- [fastgen-runtime skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md) <br>
- [fastgen-configure skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-configure/SKILL.md) <br>
- [FastGen documentation](https://docs.nvidia.com/fastgen) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Code, Configuration instructions, Analysis] <br>
**Output Format:** [Dataset files (ZIP/WebDataset shards, count manifests), Python data-config code, and a Markdown report of provenance, split/count manifest, dataset location, data-config path and observed batch contract] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive): adjacent-experiment-configuration, caption-fallback, stale-count-resume and write-prompt-shard. 1 attempt per task, each in its own isolated sandbox pod (environment: `k8s-sandbox`); each task was also attempted without the skill as a baseline. Evaluator version `1.6.8`, evaluation date 2026-11-05. <br>

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
**Overall verdict: PASS — Recommended for publication**

**Publication Recommendation:** Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.7% — baseline ran, but no comparable score was available; uplift unavailable | 91.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 90.0% → 100.0% (+10.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 68.8% — baseline ran, but no comparable score was available; uplift unavailable | 68.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 84.4% → 100.0% (+15.6 points) | 100.0% → 100.0% (±0.0 points) |
| Efficiency | 89.5% — baseline ran, but no comparable score was available; uplift unavailable | 87.3% — baseline ran, but no comparable score was available; uplift unavailable |

Baseline is the same task attempted without the target skill. Scores are rounded to one decimal; uplift is derived from the displayed scores and shown in percentage points.

### Comparison details

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.8% → 95.6% (-1.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.2% → 90.0% (-0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

Scoring: dimension bands are PASS at 50% or above, NEUTRAL from 40% to below 50%, and FAIL below 40%. The overall verdict is PASS only when every configured dimension passes for at least one supported agent.

## Skill Version(s): <br>
fbf1ad8 (source: git SHA, committed 2026-11-05) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
