## Description: <br>
Integrate a HuggingFace Computer Vision model into the NVIDIA TAO Toolkit ecosystem (tao-core config, tao-pytorch trainer, tao-deploy TensorRT pipeline). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to port HuggingFace computer-vision models (classification, detection, segmentation, depth estimation) into the NVIDIA TAO Toolkit, producing a fully integrated trainer, ONNX exporter, and TensorRT deploy pipeline. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [phase-0-prereqs.md](references/phase-0-prereqs.md) <br>
- [phase-1-inspection.md](references/phase-1-inspection.md) <br>
- [phase-2-codebase.md](references/phase-2-codebase.md) <br>
- [phase-3-implementation.md](references/phase-3-implementation.md) <br>
- [phase-4-deploy.md](references/phase-4-deploy.md) <br>
- [phase-5-packaging.md](references/phase-5-packaging.md) <br>
- [phase-6-container-tests.md](references/phase-6-container-tests.md) <br>
- [phase-7-optimization.md](references/phase-7-optimization.md) <br>
- [cross-cutting.md](references/cross-cutting.md) <br>
- [tao-patterns.md](references/tao-patterns.md) <br>
- [TAO Skill Bank (GitHub)](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive), each attempt run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (goal completion and expected workflow adherence, equal weight). <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 23% → 100% (+76 points) | 38% → 58% (+20 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 0% → 100% (+100 points) | 40% → 100% (+60 points) |
| Discoverability | 0% → 100% (+100 points) | 0% → 0% (±0 points) |
| Effectiveness | 17% → 98% (+81 points) | 48% → 90% (+42 points) |
| Efficiency | 0% → 100% (+100 points) | 0% → 0% (±0 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
