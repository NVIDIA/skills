## Description: <br>
Use when authoring or validating PAIDF augmentation YAML configs, or running remote Cosmos Transfer (including Cosmos3 WSM controls), Cosmos Predict, image-edit, or image-to-video inference. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to author, validate, and run PAIDF augmentation pipelines that caption, generate, and evaluate augmented camera data through remote generative AI endpoints. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Cloud Credentials] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Configuration Schema](references/configuration-schema.md) <br>
- [Config Decision Tree](references/config-decision-tree.md) <br>
- [Pipeline Operations](references/pipeline-operations.md) <br>
- [Captioning Strategy Guide](references/captioning-strategy-guide.md) <br>
- [Evaluator Setup Guide](references/evaluator-setup-guide.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [Image Attribute Augmentation](references/image-attribute-augmentation.md) <br>
- [Event Video Generation](references/event-video-gen.md) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Shell commands, Files] <br>
**Output Format:** [Markdown with inline YAML and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
14 evaluation tasks (13 positive, 1 negative) from dataset digest sha256:35b95353d3588e053b4d8c84f9d6067e5d33e13a9789e5048e8817daac2cb64b, with 3 attempts per task in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the user's goal was achieved and the expected workflow behavior was followed. <br>
- Efficiency: Tool-call productivity and actual uncached token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Dimension | Claude Code | Codex |
|---|---:|---:|
| Overall | 92.6% | 81.1% |
| Security | 100.0% | 93.3% |
| Correctness | 95.7% | 78.7% |
| Discoverability | 92.3% | 71.8% |
| Effectiveness | 83.4% | 77.2% |
| Efficiency | 91.6% | 84.7% |

## Skill Version(s): <br>
1.2.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
