## Description: <br>
Validate and use packed sequences and long-context training in Megatron-Bridge, including offline LLM packing, collate-time VLM packing, Energon online packing, and CP constraints. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers configuring sequence packing and long-context training for LLM and VLM fine-tuning workflows in Megatron-Bridge. <br>

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
- [Packed Sequences Documentation](docs/training/packed-sequences.md) <br>
- [Performance Tuning Guide](docs/performance-guide.md) <br>
- [Hierarchical Context Parallel](docs/training/hierarchical-context-parallel.md) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 1 task (1 positive) using evaluator version 1.2.7 in local environment with Tier 3 live agent evaluation. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Whether the answer produced is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and expected workflow (goal accuracy + behavior check). <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage (routing quality and productive tool use). <br>

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
| Overall | 60% → 98% (+38 points) | 56% → 96% (+40 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 0% → 100% (+100 points) | 40% → 100% (+60 points) |
| Discoverability | 100% → 100% (±0 points) | 50% → 88% (+38 points) |
| Effectiveness | 0% → 88% (+88 points) | 40% → 94% (+54 points) |
| Efficiency | 100% → 100% (±0 points) | 50% → 100% (+50 points) |

## Skill Version(s): <br>
1.0.0+b7643bd (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
