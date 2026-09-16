## Description: <br>
Evidence-gated workflow for MoE performance optimization in Megatron Bridge. Covers measurement contracts, the Three Walls framework, parallel folding, profiling, matched A/B tuning, and final validation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers performing full MoE throughput tuning sweeps or diagnosing MoE throughput regressions after a commit or configuration change in Megatron Bridge. <br>

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
- [Scalable Training of MoE Models with Megatron Core](https://arxiv.org/abs/2603.07685) <br>
- [NVIDIA Megatron Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks and diagnostic tables] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 2 positive evaluation tasks in isolated k8s-sandbox pods, covering full MoE throughput tuning and throughput regression diagnosis. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Checks whether the skill produces correct answers against a reference answer. <br>
- Discoverability: Checks whether the right skill was loaded and activated when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow. <br>
- Efficiency: Checks whether the skill avoided wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `skill_efficiency`: Measures routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the user's goal was achieved. <br>
- `behavior_check`: Checks whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 56% → 89% (+34 points) | 65% → 99% (+34 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 40% → 50% (+10 points) | 100% → 100% (±0 points) |
| Discoverability | 50% → 100% (+50 points) | 50% → 94% (+44 points) |
| Effectiveness | 38% → 96% (+59 points) | 69% → 100% (+31 points) |
| Efficiency | 50% → 100% (+50 points) | 6% → 100% (+94 points) |

## Testing Completed: <br>
**[x] Agent Red-Teaming** <br>
**[ ] Network Security** <br>
**[ ] Product Security** <br>

## Skill Version(s): <br>
1.0.0+b7643bd (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
