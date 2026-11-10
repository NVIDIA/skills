## Description: <br>
Prepare, register or repair datasets for NVIDIA FastGen loaders: class-conditional ZIPs, WebDataset shards of images, prompts, latents, videos, audio or camera data, and KD pairs or trajectories; then verify decoded batches. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML researchers use this skill to prepare, register or repair training and evaluation datasets for NVIDIA FastGen loaders, and to verify that decoded batches match the network's expected keys, shapes, dtypes and pairing before fine-tuning or distillation. <br>

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
- [FastGen data config presets (data.py)](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/data.py) <br>
- [NVIDIA FastGen documentation](https://docs.nvidia.com/fastgen) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline code blocks, plus dataset shards, count manifests and Python data-config files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Reports provenance, split/count manifest, dataset location, data-config path and the observed batch contract] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive): adjacent-experiment-configuration, caption-fallback, stale-count-resume and write-prompt-shard; 1 attempt per task, each run in its own isolated sandbox pod (`k8s-sandbox`), with and without the skill. Evaluated 2026-11-10 with evaluator version `1.6.9`. <br>

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
**Overall verdict: PASS — Recommended for publication**

**Publication Recommendation:** Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 93.1% — baseline ran, but no comparable score was available; uplift unavailable | 89.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 75.0% → 100.0% (+25.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 90.0% → 100.0% (+10.0 points) | 95.0% → 100.0% (+5.0 points) |
| Discoverability | 77.5% — baseline ran, but no comparable score was available; uplift unavailable | 66.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 100.0% → 100.0% (±0.0 points) | 100.0% → 96.9% (-3.1 points) |
| Efficiency | 88.1% — baseline ran, but no comparable score was available; uplift unavailable | 82.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.0% → 94.8% (-1.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.8% → 90.4% (+0.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Skill Version(s): <br>
d79498f (source: git SHA, committed 2026-11-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
