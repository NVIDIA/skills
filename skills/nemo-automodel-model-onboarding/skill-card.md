## Description: <br>
Guide for onboarding new model architectures into NeMo AutoModel, including architecture discovery, implementation patterns, registration, and validation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers adding or modifying model architecture support in NeMo AutoModel, such as LLM, VLM, and MoE model files, custom layers, state-dict adapters, registry entries, and capability flags. <br>

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
- [llm-patterns.md](llm-patterns.md) <br>
- [moe-patterns.md](moe-patterns.md) <br>
- [vlm-patterns.md](vlm-patterns.md) <br>
- [capabilities-and-precision.md](capabilities-and-precision.md) <br>
- [NeMo AutoModel Documentation](https://docs.nvidia.com/nemo/automodel/latest/index.html) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 positive evaluation tasks in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Whether the answer produced is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and activated when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed expected workflow behavior. <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 49% → 98% (+49 points) | 45% → 98% (+53 points) |
| Security | 67% → 100% (+33 points) | 50% → 100% (+50 points) |
| Correctness | 60% → 100% (+40 points) | 60% → 100% (+40 points) |
| Discoverability | 48% → 100% (+52 points) | 33% → 94% (+60 points) |
| Effectiveness | 40% → 91% (+50 points) | 61% → 98% (+37 points) |
| Efficiency | 29% → 100% (+71 points) | 20% → 100% (+80 points) |

## Skill Version(s): <br>
v1.2.1+7febc6e (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
