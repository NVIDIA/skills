## Description: <br>
Runs standard or fixed-channel softmax finetuning of NV-Segment-CT VISTA3D on CT NIfTI image/label datasets, with optional MONAI-native MLflow tracking and checkpoint evidence. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers finetuning NV-Segment-CT VISTA3D models on CT NIfTI datasets for medical image segmentation, with optional experiment tracking and checkpoint validation. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Task06 Reference Details and Results](references/task06-and-results.md) <br>
- [NV-Segment-CTMR (upstream source)](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Analysis] <br>
**Output Format:** [JSON result file and model checkpoint files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces output.json with formal Dice scores, training metrics, and checkpoint recommendation; finetuned checkpoint under output directory] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 evaluation tasks (4 positive) from a versioned skill-evaluator dataset snapshot, with 3 attempts per task, each in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was selected and activated when needed, and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal accuracy + 50% expected workflow adherence). <br>
- Efficiency: Whether wasted tool calls and token usage were avoided (50% tool productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity, measuring avoidance of wasted tool calls. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.1% | 92.9% |
| Security | 100.0% → 100.0% (±0.0 points) | 75.0% → 100.0% (+25.0 points) |
| Correctness | 56.0% → 95.0% (+39.0 points) | 95.0% → 95.0% (±0.0 points) |
| Discoverability | 99.0% | 86.3% |
| Effectiveness | 35.3% → 85.4% (+50.1 points) | 80.4% → 86.7% (+6.3 points) |
| Efficiency | 81.1% | 96.3% |

## Skill Version(s): <br>
a0ef4a9 (source: git SHA, committed 2026-09-14) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
