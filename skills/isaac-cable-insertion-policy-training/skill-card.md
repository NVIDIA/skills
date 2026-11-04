## Description: <br>
Set up the standalone IsaacLabTraining package from NVIDIA-ISAAC-ROS/isaac_ros_manipulation, then launch, monitor, resume, evaluate, and tune DisplayPort insertion policies without a separate Isaac Lab source installation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and robotics engineers use this skill to install or repair the standalone Isaac Lab training package and to plan, launch, monitor, resume, evaluate, and tune reinforcement-learning campaigns for DisplayPort cable insertion policies on GPU hosts. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Inputs and boundaries](references/inputs.md) <br>
- [Omniverse EULA acceptance](references/eula.md) <br>
- [Set up or repair the standalone training package](references/setup.md) <br>
- [Preflight from the standalone package root](references/preflight.md) <br>
- [Process environment](references/environment.md) <br>
- [Session records](references/logging.md) <br>
- [Run and monitor](references/run-monitor.md) <br>
- [Tune across runs](references/tuning.md) <br>
- [Reusable supervisor](references/supervisor.md) <br>
- [NVIDIA-ISAAC-ROS/isaac_ros_manipulation (release-r1)](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_manipulation/tree/release-r1) <br>
- [Isaac ROS Manipulation training setup guide](https://nvidia-isaac-ros.github.io/manipulation/training/setup_training.html) <br>
- [Isaac ROS Manipulation DisplayPort insertion task guide](https://nvidia-isaac-ros.github.io/manipulation/training/tasks/displayport_insertion.html) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks, plus JSON session plans and generated run reports] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Writes session records (report.md, session.json, events.jsonl, per-run logs) to the skill's ignored logs/ directory; links to native Isaac Lab checkpoints, TensorBoard data, and videos rather than copying them.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
11 positive evaluation tasks covering default, short-run, PhysX-retry, and missing-exit-marker campaign planning plus fresh, blocked, no-path, path-collision, public-access, repair, and dependency-conflict setup scenarios; one attempt per task, each in an isolated sandbox pod, compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of `skill_efficiency` and `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 86.6% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Security | 100.0% → 100.0% (±0.0 points) | Not available |
| Correctness | 56.4% → 85.5% (+29.1 points) | Not available |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Effectiveness | 36.5% → 56.1% (+19.6 points) | Not available |
| Efficiency | 91.2% — baseline ran, but no comparable score was available; uplift unavailable | Not available |

## Skill Version(s): <br>
6194078 (source: git SHA, committed 2026-11-03) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
