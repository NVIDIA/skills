## Description: <br>
Guides BlueField-3 (BF3) day-1 platform bring-up over the classic RShim/BFB path — pushing a BlueField bundle with bfb-install, bringing up the host-to-DPU TMFIFO management channel, selecting DPU mode via mlxconfig, recovering a DPU after a failed push, and verifying the install. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
External DOCA operators and engineers bringing up a physical BlueField-3 DPU use this skill to push a BFB image over the RShim interface, recover a DPU that did not come back after a failed install, route mutating mode/firmware changes to the hardware-safety policy, and verify a healthy host-side and Arm-side DOCA deployment. <br>

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
- [BF3 deployment reference details](references/details.md) <br>
- [NVIDIA DOCA SDK documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline shell code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (2 positive, 1 negative), 1 attempt per task, each run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct against the reference answer? <br>
- Discoverability: Was the right skill loaded when needed and decoys avoided? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.8% (uplift unavailable) | 91.4% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 83.3% → 100.0% (+16.7 points) |
| Correctness | 66.7% → 100.0% (+33.3 points) | 93.3% → 100.0% (+6.7 points) |
| Discoverability | 100.0% (uplift unavailable) | 85.0% (uplift unavailable) |
| Effectiveness | 72.2% → 100.0% (+27.8 points) | 100.0% → 100.0% (±0.0 points) |
| Efficiency | 84.0% (uplift unavailable) | 71.8% (uplift unavailable) |

## Skill Version(s): <br>
c98e456 (source: git SHA, committed 2026-10-28) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
