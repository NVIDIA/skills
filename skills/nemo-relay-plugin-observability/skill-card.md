## Description: <br>
Use this skill when choosing or configuring NeMo Relay 0.6 or 0.7 observability through the built-in plugin, subscribers, or exporters, including raw ATOF events, ATIF trajectories, OpenTelemetry, OpenInference, or custom event handling. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers configuring observability pipelines for NeMo Relay agent runtimes, selecting and wiring exporters for ATOF, ATIF, OpenTelemetry, or OpenInference telemetry output. <br>

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
- [ATOF Event Export Reference](references/atof.md) <br>
- [ATIF Trajectory Export Reference](references/atif.md) <br>
- [OpenTelemetry Export Reference](references/opentelemetry.md) <br>
- [OpenInference Export Reference](references/openinference.md) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 20 tasks (16 positive, 4 negative) in isolated k8s-sandbox pods, 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer produced is correct against the reference answer. <br>
- Discoverability: Whether the right skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and expected workflow (goal completion + behavior adherence). <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage, measuring routing quality and productive tool use. <br>

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
| Overall | 53% → 95% (+42 points) | 66% → 92% (+26 points) |
| Security | 100% → 100% (±0 points) | 100% → 95% (-5 points) |
| Correctness | 36% → 100% (+64 points) | 78% → 90% (+12 points) |
| Discoverability | 52% → 100% (+48 points) | 61% → 94% (+33 points) |
| Effectiveness | 38% → 89% (+50 points) | 61% → 87% (+27 points) |
| Efficiency | 39% → 88% (+50 points) | 28% → 92% (+63 points) |

## Skill Version(s): <br>
f23d697 (source: git SHA, committed 2026-07-30) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
