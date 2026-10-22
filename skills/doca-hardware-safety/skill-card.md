## Description: <br>
A cross-cutting safety overlay that wraps any change touching DPU / NIC hardware state on a live system — mlxconfig writes, firmware burns, BFB reflash, mode flips, SR-IOV or device-emulation enablement, kernel boot-parameter changes, PCIe rebind/rescan, and BlueField reboot or thermal-shutdown remediation — in pre-flight inventory, out-of-band reachability, a maintenance window, replica rehearsal, rollback, and a refuse-and-escalate rule. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers, operators, and infrastructure/SRE engineers running NVIDIA DOCA on BlueField DPUs or ConnectX NICs use this skill to gate any live hardware-state change behind a disciplined workflow (pre-flight inventory, out-of-band path, maintenance window, replica rehearsal, apply, verify, rollback) so a link-breaking or firmware-committing change cannot be recommended or applied without the preconditions that prevent an unrecoverable box. <br>

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
- [NVIDIA DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [DOCA Developer Forum](https://forums.developer.nvidia.com/c/infrastructure/doca/370) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline shell command references] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), each attempt run in its own isolated sandbox pod; dataset snapshot `skill-evaluator-dataset-snapshot/1`. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct against the reference answer? <br>
- Discoverability: Was the right skill loaded when needed, and were decoy skills avoided? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected-workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>



## Evaluation Results: <br>
| Dimension | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 97.1% (uplift unavailable) | 92.7% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 70.0% → 100.0% (+30.0 points) | 70.0% → 100.0% (+30.0 points) |
| Discoverability | 100.0% (uplift unavailable) | 95.0% (uplift unavailable) |
| Effectiveness | 70.8% → 93.8% (+23.0 points) | 52.1% → 98.1% (+46.0 points) |
| Efficiency | 91.7% (uplift unavailable) | 70.5% (uplift unavailable) |

## Skill Version(s): <br>
946399d (source: git SHA, committed 2026-10-21) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
