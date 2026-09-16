## Description: <br>
Run predictions with a finetuned KERMT checkpoint on a SMILES-only CSV. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and computational chemists use this skill to run molecular property predictions on SMILES datasets using a finetuned KERMT checkpoint inside a containerized GPU environment. <br>

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
- [KERMT paper — Multitask finetuning and acceleration of chemical pretrained models](https://arxiv.org/abs/2510.12719) <br>
- [GROVER paper — Self-supervised message passing transformer on large-scale molecular data](https://arxiv.org/abs/2007.02835) <br>
- [cuik-molmaker — GPU-accelerated molecular featurization](https://github.com/NVIDIA-Digital-Bio/cuik-molmaker) <br>
- [GROVER original implementation](https://github.com/tencent-ailab/grover) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Predictions CSV with SMILES and per-target prediction columns; run manifest JSON with replay command] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), each attempted 3 times in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against reference answers. <br>
- Discoverability: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and the expected workflow behavior was followed (equal-weight mean of goal_accuracy and behavior_check). <br>
- Efficiency: Checks tool-call productivity and token usage efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability, not Efficiency). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 78.6% | 80.3% |
| Security | 100.0% → 100.0% (±0.0 points) | 70.0% → 100.0% (+30.0 points) |
| Correctness | 25.0% → 85.0% (+60.0 points) | 50.0% → 90.0% (+40.0 points) |
| Discoverability | 91.7% | 91.7% |
| Effectiveness | 25.9% → 35.0% (+9.1 points) | 22.3% → 34.4% (+12.1 points) |
| Efficiency | 81.5% | 85.4% |

## Skill Version(s): <br>
77111e0 (source: git SHA, committed 2026-09-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
