## Description: <br>
Guides developers through adding DOCA capabilities — most commonly DOCA Flow hardware steering — to an existing DPDK application in-place using the DOCA DPDK Bridge, covering DPDK port binding to a `doca_dev`, `rte_mbuf` ↔ `doca_buf` conversion, capability queries, and debugging `DOCA_ERROR_*` returned from bridge calls. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
External developers who already maintain a DPDK-based packet-processing application use this skill to add DOCA capabilities (most commonly hardware steering via DOCA Flow) to that application in-place, binding DPDK ports to DOCA devices, converting buffers at the data-plane boundary, checking DPDK ↔ DOCA version compatibility, and debugging bridge errors without rewriting the data-plane in DOCA-native form. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NVIDIA DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [NVIDIA DOCA Samples](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [NVIDIA DOCA Platform Framework](https://github.com/NVIDIA/doca-platform) <br>
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
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each run in its own isolated k8s-sandbox pod against dataset snapshot skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the final answer correct against the reference answer? <br>
- Discoverability: Was the right skill loaded when needed and were decoys avoided? <br>
- Effectiveness: Did the skill help complete the task, measured as the equal-weight mean of goal completion and expected workflow adherence? <br>
- Efficiency: Did it avoid wasted tool calls and token usage, measured as tool-call productivity and token efficiency? <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.1% — baseline ran, but no comparable score was available; uplift unavailable | 89.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 100.0% (+12.5 points) |
| Correctness | 70.0% → 100.0% (+30.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.0% → 95.8% (+45.8 points) | 54.2% → 83.3% (+29.1 points) |
| Efficiency | 84.8% — baseline ran, but no comparable score was available; uplift unavailable | 69.3% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
c98e456 (source: git SHA, committed 2026-10-28) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
