## Description: <br>
Use this skill when the operator is driving the DOCA Socket Relay to bridge a socket-oriented host application onto a BlueField DPU peer without rewriting it — picking the deployment shape (in-process, sidecar, or BlueField service container), configuring the host-side socket and the DPU-side forwarding endpoint, walking the bind → connect → round-trip → admit-fleet smoke, or diagnosing a stuck/silent relay. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers, application owners, migration engineers, and SREs use this skill to bridge a socket-oriented host application onto a BlueField DPU without rewriting it against the DOCA programming surface — selecting the deployment shape, configuring the relay's host-side socket and DPU-side forwarding endpoint, running the smoke-before-bulk workflow, and diagnosing relays that refuse connections or silently drop data. <br>

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
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>
- [DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [DOCA Samples](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [DOCA Platform Framework](https://github.com/NVIDIA/doca-platform) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), 1 attempt per task, each run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (scored from the `security` signal). <br>
- Correctness: Whether the answer is correct (scored from the `accuracy` signal). <br>
- Discoverability: Whether the right skill was loaded when needed (scored from the `skill_execution` signal). <br>
- Effectiveness: Whether the skill helped complete the task — equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage — 50% tool-call productivity (`skill_efficiency`) and 50% `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.4% — baseline ran, but no comparable score was available; uplift unavailable | 94.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 55.0% → 100.0% (+45.0 points) | 80.0% → 100.0% (+20.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.0% → 93.8% (+43.8 points) | 58.8% → 86.3% (+27.5 points) |
| Efficiency | 68.1% — baseline ran, but no comparable score was available; uplift unavailable | 95.8% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
80aaaff (source: git SHA, committed 2026-10-13) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
