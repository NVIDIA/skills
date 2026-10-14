## Description: <br>
Use this skill for BlueField-3 (BF3) day-1 platform bring-up via the classic RShim/BFB path: pushing a BlueField bundle (BFB) to the DPU over RShim with bfb-install from the host, the host-to-DPU TMFIFO management channel, RShim daemon state and console-over-rshim, DPU mode selection via mlxconfig, post-BFB recovery, a six-state BlueField-state classifier, and verifying the install. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and operators performing BlueField-3 day-1 platform bring-up, BFB image deployment over RShim, DPU recovery, and install verification on physical BF3 hardware. <br>

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
- [BF3 deployment details](references/details.md) <br>
- [NVIDIA DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [DOCA Samples](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [DOCA Platform Framework](https://github.com/NVIDIA/doca-platform) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (2 positive, 1 negative), 1 attempt per task, each in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct — final-answer correctness against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed — skill selection, decoy avoidance, and workflow execution. <br>
- Effectiveness: Whether the skill helped complete the task — goal completion (50%) and expected workflow adherence (50%). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage — tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 98.0% | 95.9% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 80.0% → 100.0% (+20.0 points) | 73.3% → 100.0% (+26.7 points) |
| Discoverability | 100.0% | 90.0% |
| Effectiveness | 76.1% → 100.0% (+23.9 points) | 60.0% → 98.3% (+38.3 points) |
| Efficiency | 89.7% | 91.1% |

## Skill Version(s): <br>
d2dda0c (source: git SHA, committed 2026-10-12) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
