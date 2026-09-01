## Description: <br>
Used for converting one CT DICOM series folder to a HU NIfTI volume with affine evidence. Not for multi-frame DICOM or clinical use. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to convert single-series CT DICOM directories into HU-scaled NIfTI volumes with affine geometry for engineering verification and development workflows. <br>

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
- [BENCHMARK.md](BENCHMARK.md) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Analysis] <br>
**Output Format:** [NIfTI volume (.nii.gz) with JSON summary] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Key output fields: n_slices, series_instance_uid, output.path, output.shape, output.spacing, output.axcodes, output.affine, hu_range, runtime.conversion_seconds] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
2 evaluation tasks (1 positive, 1 negative) per agent, each in an isolated sandbox pod. Dataset: skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use: checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct: final-answer correctness against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed: checks skill execution and routing. <br>
- Effectiveness: Whether the skill helped complete the task: equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Whether wasted tool or skill usage was avoided: routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 70% → 90% (+20 points) | 83% → 93% (+10 points) |
| Security | 100% → 100% (±0 points) | 50% → 100% (+50 points) |
| Correctness | 50% → 80% (+30 points) | 100% → 100% (±0 points) |
| Discoverability | 75% → 94% (+19 points) | 84% → 84% (±0 points) |
| Effectiveness | 56% → 79% (+22 points) | 92% → 90% (-2 points) |
| Efficiency | 70% → 97% (+27 points) | 89% → 90% (+1 points) |

## Skill Version(s): <br>
0.1.0 (source: skill_manifest.yaml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
