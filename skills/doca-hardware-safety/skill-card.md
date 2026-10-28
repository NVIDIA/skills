## Description: <br>
A cross-cutting safety overlay that wraps any change touching DPU / NIC hardware state on a live system — mlxconfig writes, firmware burns, BFB reflash, NIC ↔ DPU mode flips, SR-IOV or device-emulation enablement, kernel boot-parameter changes, PCIe rebind/rescan/link-state flips, and BlueField cold reboot or thermal-shutdown remediation — in pre-flight inventory, out-of-band reachability, a maintenance window, replica rehearsal, rollback, and a refuse-and-escalate rule. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers, operators, and infrastructure/SRE engineers running NVIDIA DOCA on BlueField DPUs or ConnectX NICs use this skill to gate any live hardware-state change behind a disciplined workflow (pre-flight inventory, out-of-band path, maintenance window, replica rehearsal, apply, verify, rollback) so a link-breaking or firmware-committing change cannot be recommended or applied without the preconditions that prevent an unrecoverable system. <br>

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
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each attempt run in its own isolated sandbox pod; dataset snapshot `skill-evaluator-dataset-snapshot/1`. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct against the reference answer? <br>
- Discoverability: Was the right skill loaded when needed, and were decoy skills avoided? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected-workflow adherence. <br>
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
| Dimension | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.4% (uplift unavailable) | 83.9% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 55.0% → 95.0% (+40.0 points) | 50.0% → 75.0% (+25.0 points) |
| Discoverability | 100.0% (uplift unavailable) | 95.0% (uplift unavailable) |
| Effectiveness | 39.6% → 93.8% (+54.2 points) | 68.8% → 81.3% (+12.5 points) |
| Efficiency | 93.3% (uplift unavailable) | 68.4% (uplift unavailable) |

## Skill Version(s): <br>
c98e456 (source: git SHA, committed 2026-10-28) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
