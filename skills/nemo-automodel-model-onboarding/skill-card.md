## Description: <br>
Guide for onboarding new model architectures into NeMo AutoModel, including architecture discovery, implementation patterns, registration, and validation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers adding new LLM, VLM, or MoE model architecture support to NeMo AutoModel, including HuggingFace config analysis, model file implementation, state-dict adapter authoring, registry registration, and validation testing. <br>

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
**Output Format:** [Markdown with inline Python and YAML code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 internal evaluation tasks (3 positive) in isolated k8s-sandbox pods. Dataset digest: sha256:814bfc7c94da8ea6fd1a065d7f3f0c4fcda9ef918f1da7eb46630700f241fc25. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, based on unsafe operations, secret leakage, and unauthorized access checks. <br>
- Correctness: Whether the skill produces correct answers, based on final-answer accuracy against reference answers. <br>
- Discoverability: Whether the right skill was loaded when needed, based on skill execution and routing quality. <br>
- Effectiveness: Whether the skill helps complete the user's goal, combining goal completion (50%) and expected workflow adherence (50%). <br>
- Efficiency: Whether the skill avoids wasted tool calls and token usage, combining tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity; routing is scored under Discoverability. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code | Codex |
|---|---:|---:|
| Overall | 98.5% | 97.8% |
| Security | 100.0% | 100.0% |
| Correctness | 100.0% | 100.0% |
| Discoverability | 100.0% | 91.7% |
| Effectiveness | 92.6% | 97.2% |
| Efficiency | 100.0% | 100.0% |

## Skill Version(s): <br>
v1.2.1+10d4dd7 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
