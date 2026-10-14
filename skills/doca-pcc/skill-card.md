## Description: <br>
Guides host-side loading, parameterization, capability discovery, and debugging of a custom Programmable Congestion Control algorithm on a BlueField DPU using the DOCA PCC library. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers deploying custom Programmable Congestion Control algorithms onto BlueField DPUs, using the host-side DOCA PCC library to create per-port contexts, load DPACC-compiled algorithm images, parameterize them, query capabilities, and debug DOCA_ERROR codes. <br>

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
- [DOCA PCC SDK Documentation](https://docs.nvidia.com/doca/sdk/DOCA-PCC/index.html) <br>
- [DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [DOCA Samples (GitHub)](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [DOCA Platform Framework (GitHub)](https://github.com/NVIDIA/doca-platform) <br>
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions, Shell commands, Code] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 tasks (3 positive, 1 negative) from skill-evaluator-dataset-snapshot/1, with 1 attempt per task in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Measures final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded when needed. <br>
- Effectiveness: Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: 50% tool-call productivity and 50% token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Dimension | # Tasks | Claude Code | Codex |
|---|---:|---:|---:|
| Overall | 4 | 93.4% | 94.8% |
| Security | 4 | 100.0% | 100.0% |
| Correctness | 4 | 100.0% | 100.0% |
| Discoverability | 4 | 100.0% | 93.3% |
| Effectiveness | 4 | 73.1% | 95.6% |
| Efficiency | 4 | 93.9% | 85.2% |

## Skill Version(s): <br>
00a43cf (source: git SHA, committed 2026-10-11) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
