## Description: <br>
Evaluate models, datasets, and agents with the NeMo Evaluator plugin for metric selection, SDK checks, platform jobs, and result retrieval. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers evaluating LLM and agent performance using NeMo Platform's evaluation framework for metric selection, job submission, and result retrieval. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Agent Evaluation](references/agent-evaluation.md) <br>
- [API Auth](references/api-auth.md) <br>
- [Evaluation Shapes](references/evaluation-shapes.md) <br>
- [Execution](references/execution.md) <br>
- [LLM Judge](references/llm-judge.md) <br>
- [Metric Selection](references/metric-selection.md) <br>
- [Resources](references/resources.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, API Calls, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive) from skill-evaluator-dataset-snapshot, evaluated in k8s-sandbox environment with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? <br>
- Correctness: Is the answer correct? <br>
- Discoverability: Was the right skill loaded when needed? <br>
- Effectiveness: Did the skill help complete the task? <br>
- Efficiency: Did it avoid wasted tool or skill usage? <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill) | Codex (Baseline → Skill) |
|---|---:|---:|
| Overall | 45% → 95% (+50) | 52% → 90% (+38) |
| Security | 100% → 100% (±0) | 50% → 100% (+50) |
| Correctness | 40% → 100% (+60) | 60% → 80% (+20) |
| Discoverability | 44% → 100% (+56) | 50% → 88% (+38) |
| Effectiveness | 10% → 75% (+65) | 75% → 85% (+10) |
| Efficiency | 30% → 100% (+70) | 25% → 100% (+75) |

## Skill Version(s): <br>
acfad8e4 (source: git SHA, committed 2026-08-10) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
