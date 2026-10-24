## Description: <br>
Deploy, tune, and evaluate NVIDIA's DOCA-shipped Zero-Touch RoCE RTT congestion control (ZTR RTTCC) reference algorithm (`doca_pcc_dev_ztr_rttcc_algo`) on a BlueField-3 DPA through the shipped DOCA PCC sample, including variant selection at DPACC build time, host-set parameter tuning, and `DOCA_PCC_DEV_STATUS_FAIL` debugging. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers operating a BlueField-3-class DPU use this skill to deploy, tune, and evaluate the DOCA-shipped ZTR RTTCC reference congestion-control algorithm on a port carrying RoCE-v2 traffic, including deciding whether the shipped algorithm is the right baseline versus a custom algorithm. <br>

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
- [Capabilities companion (CAPABILITIES.md)](CAPABILITIES.md) <br>
- [Tasks companion (TASKS.md)](TASKS.md) <br>
- [DOCA PCC Programming Guide](https://docs.nvidia.com/doca/sdk/doca-pcc/index.html) <br>
- [DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline bash and C code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and activated when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal and the expected workflow was followed. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Flags unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Measures tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Checks final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the user's goal was achieved. <br>
- `behavior_check`: Checks whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.0% (uplift unavailable) | 88.8% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 75.0% → 100.0% (+25.0 points) |
| Discoverability | 100.0% (uplift unavailable) | 95.0% (uplift unavailable) |
| Effectiveness | 48.8% → 78.7% (+29.9 points) | 74.3% → 86.9% (+12.6 points) |
| Efficiency | 76.2% (uplift unavailable) | 62.1% (uplift unavailable) |

## Skill Version(s): <br>
b99bced (source: git SHA, committed 2026-10-23) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
