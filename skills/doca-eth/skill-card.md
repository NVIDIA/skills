## Description: <br>
Guides hands-on DOCA Ethernet packet-queue work on a BlueField DPU or ConnectX NIC: bringing up doca_eth_rxq and doca_eth_txq contexts on a port, representor, or SF, choosing among the regular, cyclic, managed-mempool, and shared-mempool RX types, sizing burst and scatter-gather length against the _cap_* queries, submitting doca_eth_txq_task_send / _lso_send tasks with packet doca_bufs, and debugging DOCA_ERROR_* from Ethernet calls. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
External developers and engineers building applications that consume the DOCA Ethernet library use this skill to set up RX and TX queues as independent Core contexts, pick an RX type and query device capabilities, send packets through doca_eth_txq, find out why an RX queue gets no traffic, and work out which layer caused an Ethernet-related DOCA error on BlueField or ConnectX devices. <br>

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
- [DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [NVIDIA DOCA Samples](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash and C code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each run in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is the skill safe to use (unsafe operations, secret leakage, and unauthorized access). <br>
- Correctness: Is the final answer correct against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed and were decoys avoided. <br>
- Effectiveness: Did the skill help complete the task (equal-weight mean of goal completion and expected workflow adherence). <br>
- Efficiency: Did it avoid wasted tool calls and token usage (tool-call productivity and token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.7% — baseline ran, but no comparable score was available; uplift unavailable | 91.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 50.0% → 100.0% (+50.0 points) | 75.0% → 95.0% (+20.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 45.8% → 100.0% (+54.2 points) | 52.1% → 100.0% (+47.9 points) |
| Efficiency | 83.6% — baseline ran, but no comparable score was available; uplift unavailable | 69.1% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
c98e456 (source: git SHA, committed 2026-10-28) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
