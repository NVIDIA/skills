## Description: <br>
Guides deployment and configuration of the NVIDIA DOCA Telemetry Service (DTS) across its three deployment shapes (BlueField built-in static pod, host docker container, or DPF DPUService helm chart) and their shared dts_config.ini layer of providers, exporters, and level labels, routing DOCA install, telemetry library API, DPF install, Grafana/Prometheus server, and BFB/BSP imaging questions to other skills. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and operators use this skill to deploy, configure, upgrade, and debug DOCA Telemetry Service on BlueField DPUs, docker hosts, or DPF-managed clusters, including enabling providers, choosing exporters, attaching level labels, and verifying that metrics are actually exported. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Other [root/sudo access or Kubernetes cluster credentials]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>
- [DOCA Telemetry Service Guide](https://docs.nvidia.com/doca/sdk/doca+telemetry+service+guide/index.html) <br>
- [DOCA Container Deployment Guide](https://docs.nvidia.com/doca/sdk/doca+container+deployment+guide/index.html) <br>
- [NGC doca_telemetry Container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/doca/containers/doca_telemetry) <br>
- [NGC doca-telemetry Helm Chart](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/doca/helm-charts/doca-telemetry) <br>
- [DOCA Platform Framework (DPF) Documentation](https://docs.nvidia.com/networking/display/dpf/) <br>
- [NVIDIA DOCA Platform (GitHub)](https://github.com/NVIDIA/doca-platform) <br>
- [NVIDIA DOCA Developer Forum](https://forums.developer.nvidia.com/c/infrastructure/doca/370) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative), one attempt per task, each in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was loaded when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the task, as the equal-weight mean of goal completion and expected-workflow adherence. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage, as the equal-weight mean of tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline -> Skill Uplift) | Codex (Baseline -> Skill Uplift) |
|---|---:|---:|
| Overall | 86.2% (uplift unavailable) | 79.4% (uplift unavailable) |
| Security | 100.0% -> 100.0% (±0.0 points) | 100.0% -> 83.3% (-16.7 points) |
| Correctness | 60.0% -> 80.0% (+20.0 points) | 76.7% -> 80.0% (+3.3 points) |
| Discoverability | 100.0% (uplift unavailable) | 95.0% (uplift unavailable) |
| Effectiveness | 29.2% -> 74.3% (+45.1 points) | 36.1% -> 75.0% (+38.9 points) |
| Efficiency | 76.7% (uplift unavailable) | 63.8% (uplift unavailable) |

## Skill Version(s): <br>
c98e456 (source: git SHA, committed 2026-10-28) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
