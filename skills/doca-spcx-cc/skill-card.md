## Description: <br>
Guides invoking the host-side doca_spcx_cc CLI to load, parameterize, start, observe, and stop an SPCX-class Programmable Congestion Control algorithm on a BlueField with a DPA processor against a live RDMA / RoCE fabric, including the SPCX-vs-doca-pcc-vs-factory-firmware decision, contention-positive evaluation, layered error diagnosis, and the safety gate before production rollout. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers, CC researchers, and platform operators use this skill to choose between SPCX, doca-pcc, and factory PCC, then load, run, and evaluate a custom or shipped reference congestion-control algorithm with doca_spcx_cc on a non-prod RDMA / RoCE fabric under controlled contention, diagnose errors such as DOCA_PCC_PS_ERROR, and decide whether the algorithm is safe to roll forward with documented evidence and a factory-PCC rollback plan. <br>

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
- [doca-pcc skill](../../libs/doca-pcc/SKILL.md) <br>
- [doca-pcc-ztr-rttcc-algo skill](../../libs/doca-pcc-ztr-rttcc-algo/SKILL.md) <br>
- [doca-dpa skill](../../libs/doca-dpa/SKILL.md) <br>
- [doca-hardware-safety skill](../../doca-hardware-safety/SKILL.md) <br>
- [doca-version skill](../../doca-version/SKILL.md) <br>
- [DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Measures final-answer accuracy against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? Measures whether the expected skill was selected and decoys were avoided. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Combines tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity; routing is scored under Discoverability. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 94.2% — baseline ran, but no comparable score was available; uplift unavailable | 94.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 75.0% → 100.0% (+25.0 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 65.0% → 100.0% (+35.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 86.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.0% → 100.0% (+50.0 points) | 38.8% → 100.0% (+61.2 points) |
| Efficiency | 71.1% — baseline ran, but no comparable score was available; uplift unavailable | 86.7% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
7e143d8 (source: git SHA, committed 2026-10-12) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
