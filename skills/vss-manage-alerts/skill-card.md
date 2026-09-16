## Description: <br>
Use for VSS alert workflows — real-time monitoring, Alert-Bridge subscriptions, Slack notifications, incident queries, camera onboarding. Not for non-alert analytics. <br>

This skill is for demonstration purposes and not for production usage. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 OR MIT <br>
## Use Case: <br>
Developers and engineers operating the NVIDIA Video Search and Summarization AI Blueprint alert pipeline — managing real-time monitoring, alert subscriptions, Slack notifications, incident queries, and camera onboarding. <br>

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
- [Alert Notify Reference](references/alert-notify.md) <br>
- [Alert Subscriptions Reference](references/alert-subscriptions.md) <br>
- [CV Verifier Prompts Reference](references/cv-verifier-prompts.md) <br>
- [NVIDIA Video Search and Summarization GitHub](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) <br>
- [VSS Documentation](https://docs.nvidia.com/vss/latest/index.html) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, API Calls, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (6 positive, 1 negative) against skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 51% → 90% (+39 points) | 50% → 80% (+30 points) |
| Security | 100% → 86% (-14 points) | 86% → 64% (-21 points) |
| Correctness | 20% → 100% (+80 points) | 26% → 91% (+66 points) |
| Discoverability | 56% → 100% (+43 points) | 51% → 87% (+36 points) |
| Effectiveness | 27% → 70% (+43 points) | 37% → 65% (+27 points) |
| Efficiency | 51% → 93% (+43 points) | 53% → 94% (+42 points) |

## Skill Version(s): <br>
3.2.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
