## Description: <br>
Guide for onboarding new model architectures into NeMo AutoModel, including architecture discovery, implementation patterns, registration, and validation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers adding or modifying model architecture support in NeMo AutoModel, including LLM, VLM, and MoE model files, custom layers, state-dict adapters, registry entries, Hugging Face config mapping, and capability flags. <br>

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
- [LLM Patterns](llm-patterns.md) <br>
- [MoE Patterns](moe-patterns.md) <br>
- [VLM Patterns](vlm-patterns.md) <br>
- [Capabilities and Precision](capabilities-and-precision.md) <br>
- [NeMo AutoModel Documentation](https://docs.nvidia.com/nemo/automodel/latest/index.html) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 tasks (3 positive) from a pinned dataset snapshot, each attempt in an isolated sandbox pod. Tasks cover dense LLM onboarding, MoE state-dict adapter mapping, and VLM onboarding. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed — skill selection, decoy avoidance, and workflow execution. <br>
- Effectiveness: Whether the skill helped complete the task — equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage — 50% tool-call productivity and 50% token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Skill) | Codex (Skill) |
|---|---:|---:|
| Overall | 98.2% | 95.8% |
| Security | 100.0% | 100.0% |
| Correctness | 100.0% | 100.0% |
| Discoverability | 100.0% | 91.7% |
| Effectiveness | 93.5% | 89.2% |
| Efficiency | 97.2% | 98.2% |

## Skill Version(s): <br>
v1.2.1+4214430 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
