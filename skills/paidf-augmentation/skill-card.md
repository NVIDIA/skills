## Description: <br>
Use when authoring or validating PAIDF augmentation YAML configs, or running remote Cosmos Transfer/Predict, image-edit, or image-to-video inference. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to drive the PAIDF augmentation pipeline — selecting generative-AI models, authoring and validating YAML configs, configuring captioning and evaluators, and launching remote inference for video and image augmentation. <br>

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
- [configuration-schema.md](references/configuration-schema.md) <br>
- [config-decision-tree.md](references/config-decision-tree.md) <br>
- [pipeline-operations.md](references/pipeline-operations.md) <br>
- [captioning-strategy-guide.md](references/captioning-strategy-guide.md) <br>
- [evaluator-setup-guide.md](references/evaluator-setup-guide.md) <br>
- [troubleshooting.md](references/troubleshooting.md) <br>
- [image-attribute-augmentation.md](references/image-attribute-augmentation.md) <br>
- [event-video-gen.md](references/event-video-gen.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash and YAML code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 evaluation tasks (12 positive, 1 negative) run in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the user's goal was achieved and expected workflow behavior was followed. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

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
| Overall | 52% → 93% (+41 points) | 49% → 86% (+37 points) |
| Security | 100% → 100% (±0 points) | 73% → 92% (+19 points) |
| Correctness | 45% → 100% (+55 points) | 49% → 91% (+42 points) |
| Discoverability | 42% → 92% (+50 points) | 48% → 79% (+31 points) |
| Effectiveness | 41% → 88% (+47 points) | 40% → 88% (+48 points) |
| Efficiency | 33% → 83% (+50 points) | 36% → 81% (+45 points) |

## Skill Version(s): <br>
1.1.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
