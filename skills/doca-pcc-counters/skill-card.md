## Description: <br>
Guides an AI agent through arming and reading the fixed firmware/hardware PCC (Programmable Congestion Control) diagnostic counters (CNP, RTT, WRED-drop, etc.) on a ConnectX / BlueField device using the `pcc_counters.sh` script via the mlx5 debugfs `diag_cnt` interface. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers, operators, and AI agents use this skill to arm and read firmware/hardware PCC diagnostic counters on ConnectX / BlueField devices to investigate congestion-control behavior such as CNP generation, RTT requests/responses, and WRED drops. <br>

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
- [DOCA Samples (GitHub)](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [DOCA Platform Framework (GitHub)](https://github.com/NVIDIA/doca-platform) <br>
- [DOCA Developer Forum](https://forums.developer.nvidia.com/c/infrastructure/doca/370) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 evaluation tasks (3 positive, 1 negative) with 3 attempts per task in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use — no unsafe operations, secret leakage, or unauthorized access. <br>
- Correctness: Checks whether the final answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded and activated when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow (50% goal completion + 50% behavior adherence). <br>
- Efficiency: Checks whether the skill avoided wasted tool calls and token usage (50% tool productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability, not Efficiency). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 95.9% | 98.3% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 23.3% → 100.0% (+76.7 points) | 80.0% → 100.0% (+20.0 points) |
| Discoverability | 100.0% | 95.0% |
| Effectiveness | 41.7% → 87.5% (+45.8 points) | 30.0% → 100.0% (+70.0 points) |
| Efficiency | 92.0% | 96.5% |

## Skill Version(s): <br>
be5fb89 (source: git SHA, committed 2026-09-14) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
