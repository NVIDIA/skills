## Description: <br>
Bootstrap the KERMT agent environment — verify host docker + nvidia-container-toolkit, build the kermt:latest image from the repo’s Dockerfile if it doesn’t yet exist, and run a GPU smoke test inside the container. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to bootstrap a containerized KERMT environment for molecular property prediction model training, finetuning, and inference workflows. <br>

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
- [KERMT paper (Multitask finetuning and acceleration of chemical pretrained models)](https://arxiv.org/abs/2510.12719) <br>
- [GROVER paper (Self-Supervised Message Passing Transformer)](https://arxiv.org/abs/2007.02835) <br>
- [cuik-molmaker (NVIDIA Digital Bio)](https://github.com/NVIDIA-Digital-Bio/cuik-molmaker) <br>
- [GROVER original implementation](https://github.com/tencent-ailab/grover) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 tasks (3 positive, 1 negative) with 3 attempts each, executed in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the skill produces correct final answers against reference outputs. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the skill helped complete the user’s goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `goal_accuracy`: Whether the user’s goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity measured against baseline. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 80.5% | 89.1% |
| Security | 100.0% → 50.0% (-50.0 pts) | 50.0% → 100.0% (+50.0 pts) |
| Correctness | 40.0% → 100.0% (+60.0 pts) | 60.0% → 100.0% (+40.0 pts) |
| Discoverability | 90.0% | 92.7% |
| Effectiveness | 31.9% → 78.8% (+46.9 pts) | 40.0% → 70.0% (+30.0 pts) |
| Efficiency | 83.5% | 82.6% |

## Skill Version(s): <br>
77111e0 (source: git SHA, committed 2026-09-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
