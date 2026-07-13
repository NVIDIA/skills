## Description: <br>
Integrate TileGym kernels into Hugging Face transformers models by replacing the library's submodule(s) and certain class(es)' implementations, and patching certain class(es)' init/forward/load weight methods prior to instantiating models. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers who need to integrate TileGym CUDA Tile kernels into Hugging Face transformers models for LLM training and inference performance optimization. <br>

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
- [Auto Kernelize](references/auto-kernelize.md) <br>
- [Environment Setup](references/environment-setup.md) <br>
- [Kernel Integration](references/kernel-integration.md) <br>
- [Kernel Inventory Schema](references/kernel-inventory-schema.md) <br>
- [Workflow Diagram](references/workflow-diagram.png) <br>
- [CUDA Tile IR Documentation](https://docs.nvidia.com/cuda/tile-ir/latest/) <br>
- [FlashInfer-Bench Definition Schema](https://github.com/flashinfer-ai/flashinfer-bench/blob/main/docs/flashinfer-trace/definition.mdx) <br>
- [FlashInfer-Bench Solution Schema](https://github.com/flashinfer-ai/flashinfer-bench/blob/main/docs/flashinfer-trace/solution.mdx) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 5 tasks (1 positive activation, 4 negative activation) in the NVSkills-Eval external profile. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 5 | 100% (+0%) | 100% (+0%) |
| Correctness | 5 | 100% (+20%) | 99% (+14%) |
| Discoverability | 5 | 100% (+20%) | 99% (+9%) |
| Effectiveness | 5 | 100% (+20%) | 99% (+17%) |
| Efficiency | 5 | 96% (+13%) | 97% (+7%) |

## Skill Version(s): <br>
2026.06.03 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
