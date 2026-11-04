## Description: <br>
Export a trained Isaac Lab DisplayPort task-space checkpoint with LEAPP, run it through Isaac ROS Deploy and Isaac Sim 6.0 in separate Docker containers, verify training asset parity, and record insertion video and measured results. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and robotics engineers use this skill for post-training simulation deployment and testing of Isaac Lab DisplayPort cable insertion policies, producing a validated LEAPP export bundle, a recorded Isaac Sim insertion trial, and measured results. <br>

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
- [NVIDIA DisplayPort simulation tutorial](https://nvidia-isaac-ros.github.io/manipulation/reference_workflows/isaac_for_manipulation/tutorials/isaac_sim/tutorial_displayport_task_space_isaac_sim.html) <br>
- [Training and export](https://nvidia-isaac-ros.github.io/manipulation/training/tasks/displayport_insertion.html) <br>
- [Isaac ROS 5.0 setup](https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/index.html) <br>
- [Setup](references/setup.md) <br>
- [Interfaces and asset parity](references/interfaces.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [Dependency inventory](references/dependencies.md) <br>
- [Docker workflow validation](evals/validation.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [Markdown report with machine-readable JSON/JSONL evidence, ONNX/LEAPP export bundle, logs, and MP4 video] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [One report.md per invocation under a unique output directory; insertion success is reported only when the complete attempt meets the saved task's success criterion] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), 1 attempt per task, each in an isolated k8s-sandbox pod; each task was also attempted without the skill as a baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` and 50% `token_efficiency`. <br>

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
| Overall | 87.9% — baseline ran, but no comparable score was available; uplift unavailable | 74.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 50.0% → 95.0% (+45.0 points) | 30.0% → 70.0% (+40.0 points) |
| Discoverability | 97.5% — baseline ran, but no comparable score was available; uplift unavailable | 66.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 40.9% → 65.0% (+24.1 points) | 40.9% → 45.0% (+4.1 points) |
| Efficiency | 82.0% — baseline ran, but no comparable score was available; uplift unavailable | 90.3% — baseline ran, but no comparable score was available; uplift unavailable |

Overall verdict: PASS (Tier 1 static validation passed with observations; Tier 2 semantic deduplication passed; Tier 3 live agent evaluation passed).

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
