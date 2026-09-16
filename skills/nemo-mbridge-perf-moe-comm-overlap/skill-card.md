## Description: <br>
MoE expert-parallel communication overlap in Megatron Bridge. Covers dispatch/combine overlap, flex dispatcher backends, and expert wgrad scheduling. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers tuning MoE communication overlap in Megatron Bridge, or tracing MoE throughput regressions to comm-overlap configuration changes. <br>

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
- [Communication Overlap Documentation](docs/training/communication-overlap.md) <br>
- [Qwen3 30B-A3B Model Verification Card](examples/model_verification_cards/qwen3-30b-a3b/card.yaml) <br>
- [Qwen3 MoE H100 Performance Recipe](src/megatron/bridge/perf_recipes/qwen/h100/qwen3_moe.py) <br>
- [Parallelisms Documentation](docs/parallelisms.md) <br>
- [Performance Tuning Guide](docs/performance-guide.md) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Analysis, Shell commands] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 1 task (1 positive) in isolated k8s-sandbox pods with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use, including unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the skill produces correct answers against a reference answer. <br>
- Discoverability: Checks whether the right skill is found and executed when needed. <br>
- Effectiveness: Checks whether the skill helps complete the user's goal and expected workflow. <br>
- Efficiency: Checks whether the skill avoids wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 37% → 97% (+61 points) | 40% → 96% (+56 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 0% → 100% (+100 points) | 0% → 100% (+100 points) |
| Discoverability | 50% → 100% (+50 points) | 50% → 94% (+44 points) |
| Effectiveness | 0% → 87% (+87 points) | 0% → 87% (+87 points) |
| Efficiency | 33% → 100% (+67 points) | 50% → 100% (+50 points) |

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
