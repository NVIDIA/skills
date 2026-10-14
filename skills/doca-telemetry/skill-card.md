## Description: <br>
Guides reading DOCA hardware-counter events from a doca_dev through the per-domain DOCA Telemetry reader libraries (doca_telemetry_pcc, _dpa, _diag, _adp_retx, _phy, and _pci), covering capability checks, context creation, startup, and per-domain reads or samples, while routing publishing/export to doca-telemetry-exporter and the deployed DOCA Telemetry Service to doca-dts. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers building applications on NVIDIA BlueField DPUs or ConnectX NICs use this skill to configure, build, run, test, and debug code that reads per-domain hardware counters (PCC, DPA, DIAG, ADP_RETX, PHY, PCI) from a doca_dev via the DOCA Telemetry reader C API or language bindings over it. <br>

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
- [references/details.md](references/details.md) <br>
- [DOCA Telemetry Guide](https://docs.nvidia.com/doca/sdk/DOCA-Telemetry/index.html) <br>
- [DOCA Telemetry Exporter Guide](https://docs.nvidia.com/doca/sdk/DOCA-Telemetry-Exporter/index.html) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline C and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative) executed in isolated k8s-sandbox pods with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and the expected workflow behavior was followed. <br>
- Efficiency: Checks tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 98.1% | 92.1% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 50.0% → 100.0% (+50.0 points) | 80.0% → 100.0% (+20.0 points) |
| Discoverability | 100.0% | 90.0% |
| Effectiveness | 51.3% → 98.8% (+47.5 points) | 56.3% → 98.1% (+41.8 points) |
| Efficiency | 91.7% | 72.4% |

## Skill Version(s): <br>
e2bc144 (source: git SHA, committed 2026-10-14) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
