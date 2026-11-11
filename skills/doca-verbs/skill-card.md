## Description: <br>
Raw verbs inside DOCA Core (doca-verbs): direct control of QP, CQ, PD, SRQ and address-handle objects through doca_verbs_context and doca_verbs_qp_create / doca_verbs_qp_modify / doca_verbs_cq_create, porting libibverbs code into DOCA Core, or a QP attribute, opcode or WR flag the doca-rdma task API does not expose, while routing ordinary doca-rdma / doca-eth / doca-rmax workloads, DOCA install, Core internals, and general libibverbs theory elsewhere. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
External developers and engineers building applications that consume the DOCA Verbs library (in C/C++ or via FFI/bindings) for raw QP / CQ / PD / MR / SRQ / Address-Handle / Ethernet-SQ / Ethernet-RQ control inside a DOCA Core context use this skill to decide whether dropping below doca-rdma / doca-eth / doca-rmax is actually needed, port libibverbs code into the DOCA Core model, and configure, build, modify, run, test, and debug raw-verbs applications (including diagnosing DOCA_ERROR_IO_FAILED from completion-queue entries). <br>

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
- [DOCA Samples (NVIDIA-DOCA/doca-samples)](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>
- [references/command-appendix.md](references/command-appendix.md) <br>
- [references/install.md](references/install.md) <br>
- [references/use.md](references/use.md) <br>
- [references/pattern-overview.md](references/pattern-overview.md) <br>
- [references/deferred-topics.md](references/deferred-topics.md) <br>
- [references/deferred-verbs.md](references/deferred-verbs.md) <br>
- [BENCHMARK.md](BENCHMARK.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Code, Configuration instructions] <br>
**Output Format:** [Markdown with inline C and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), 1 attempt per task, each run in its own isolated k8s-sandbox pod and compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the security signal (100%). <br>
- Correctness: Is the answer correct? Scored from final-answer accuracy (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from skill execution (100%). <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal accuracy and behavior check. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 93.8% — baseline ran, but no comparable score was available; uplift unavailable | 89.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 100.0% (+12.5 points) |
| Correctness | 45.0% → 95.0% (+50.0 points) | 80.0% → 95.0% (+15.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 31.3% → 87.5% (+56.2 points) | 68.8% → 87.5% (+18.7 points) |
| Efficiency | 86.5% — baseline ran, but no comparable score was available; uplift unavailable | 67.8% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
c98e456 (source: git SHA, committed 2026-10-28) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
