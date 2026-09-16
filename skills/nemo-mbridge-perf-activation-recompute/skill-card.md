## Description: <br>
Validate and use selective and full activation recompute in Megatron Bridge to reduce GPU memory usage at the cost of extra compute. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers diagnosing activation-memory OOMs and configuring selective or full activation recompute boundaries in Megatron Bridge training recipes. <br>

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
- [Activation Recomputation (Megatron Bridge docs)](docs/training/activation-recomputation.md) <br>
- [Megatron Core API Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/index.html) <br>
- [Performance Tuning Guide](docs/performance-guide.md) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 positive evaluation tasks covering architecture-aware memory diagnosis, matched Moonlight 16B evidence interpretation, and matched Nemotron 3 Nano capacity evidence interpretation. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (no unsafe operations, secret leakage, or unauthorized access). <br>
- Correctness: Whether the skill produces correct answers against reference ground truth. <br>
- Discoverability: Whether the right skill is loaded and activated when needed. <br>
- Effectiveness: Whether the skill helps the agent complete the user's goal and expected workflow. <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill) | Codex (Baseline → Skill) |
|---|---:|---:|
| Overall | 65% → 98% (+34 pts) | 58% → 98% (+39 pts) |
| Security | 100% → 100% (±0) | 100% → 100% (±0) |
| Correctness | 13% → 100% (+87 pts) | 73% → 100% (+27 pts) |
| Discoverability | 100% → 100% (±0) | 42% → 94% (+52 pts) |
| Effectiveness | 13% → 91% (+78 pts) | 45% → 95% (+50 pts) |
| Efficiency | 97% → 100% (+3 pts) | 32% → 100% (+68 pts) |

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
