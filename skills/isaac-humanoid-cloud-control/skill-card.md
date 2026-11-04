## Description: <br>
Run streamed Unitree G1 cloud-navigation demos in Isaac Sim with guarded gantry release. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run NVIDIA's public Isaac ROS release-4.6 Unitree G1 cloud-control workflow in Isaac Sim as an ordered integration — headless full streaming, an interactive browser WebRTC view, a guarded simulation-only gantry release, and operator-approved navigation missions through Mission Control. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Other [message broker credentials]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Release-4.6 runbook](references/release-4.6.md) <br>
- [Ordered startup and mission workflow](references/ordered-startup.md) <br>
- [Dependency workflow](references/dependencies.md) <br>
- [Fresh-machine bootstrap gate](references/bootstrap.md) <br>
- [Cloud workspace procedure](references/cloud-workspace.md) <br>
- [CLI installation and Docker setup](references/isaac-ros-setup.md) <br>
- [Readiness and troubleshooting checklist](references/troubleshooting.md) <br>
- [Teardown procedure](references/teardown.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown status report (State, Viewer, Mission, Cleanup) with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
15 evaluation tasks (15 positive), one attempt per task, each run in an isolated k8s-sandbox pod (evaluator version 1.5.6, evaluated 2026-10-22). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed, decoys avoided, and the workflow executed? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 82.3% — uplift unavailable | 68.9% — uplift unavailable |
| Security | 80.0% → 100.0% (+20.0 points) | 46.7% → 60.0% (+13.3 points) |
| Correctness | 28.0% → 78.7% (+50.7 points) | 20.0% → 72.0% (+52.0 points) |
| Discoverability | 92.3% — uplift unavailable | 80.0% — uplift unavailable |
| Effectiveness | 30.7% → 51.8% (+21.1 points) | 24.7% → 51.0% (+26.3 points) |
| Efficiency | 88.5% — uplift unavailable | 81.4% — uplift unavailable |

## Skill Version(s): <br>
0.3.0 (source: repo CHANGELOG, released 2025-05-21) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
