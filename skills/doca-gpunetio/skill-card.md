## Description: <br>
Guides hands-on DOCA GPUNetIO programming — wiring a CUDA kernel on an NVIDIA GPU to doca-eth queues via doca_gpu_eth_rxq / doca_gpu_eth_txq, standing up the per-CUDA-device doca_gpu context, designing the persistent CUDA kernel, running the dual capability check (DOCA cap-query plus cudaGetDeviceProperties), registering cudaMalloc pools via doca_buf_arr_create_*, and debugging DOCA_ERROR_* returns from the GPUNetIO API. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
External developers and engineers building applications that consume the DOCA GPUNetIO library use this skill to wire DOCA network queues into CUDA kernels on an NVIDIA GPU — standing up the per-device doca_gpu context, creating GPU-visible RX/TX queue handles on top of doca-eth, discovering GPUNetIO capabilities across DOCA and CUDA, and debugging DOCA_ERROR_* returns. <br>

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
- [DOCA GPUNetIO SDK Documentation](https://docs.nvidia.com/doca/sdk/DOCA-GPUNetIO/index.html) <br>
- [DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [NVIDIA DOCA Samples](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each run in an isolated k8s-sandbox pod against dataset snapshot skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct against the reference answer? <br>
- Discoverability: Was the right skill loaded when needed, and were decoys avoided? <br>
- Effectiveness: Did the skill help complete the task (equal-weight mean of goal completion and expected workflow adherence)? <br>
- Efficiency: Did it avoid wasted tool calls and token usage? <br>

Underlying evaluation signals used in this run: <br>
- `security`: Flags unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 89.7% (uplift unavailable) | 79.8% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 100.0% (+12.5 points) |
| Correctness | 65.0% → 95.0% (+30.0 points) | 85.0% → 95.0% (+10.0 points) |
| Discoverability | 100.0% (uplift unavailable) | 95.0% (uplift unavailable) |
| Effectiveness | 40.6% → 60.0% (+19.4 points) | 48.8% → 60.4% (+11.6 points) |
| Efficiency | 93.3% (uplift unavailable) | 48.8% (uplift unavailable) |

## Skill Version(s): <br>
b99bced (source: git SHA, committed 2026-10-23) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
