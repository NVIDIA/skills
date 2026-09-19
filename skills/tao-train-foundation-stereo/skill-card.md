## Description: <br>
Stereo depth estimation using FoundationStereo. Predicts disparity maps from stereo image pairs for 3D reconstruction. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers training, evaluating, exporting, or running inference on TAO FoundationStereo stereo depth estimation models for 3D reconstruction from stereo image pairs. <br>

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
- [Parameters — Foundation Stereo](references/parameters-foundation-stereo.md) <br>
- [Spec Overrides — Foundation Stereo](references/spec-overrides-foundation-stereo.md) <br>
- [TAO Deploy — Foundation Stereo](references/tao-deploy-foundation-stereo.md) <br>
- [Troubleshooting — Foundation Stereo](references/troubleshooting-foundation-stereo.md) <br>
- [Checkpoint Inference Mappings — Foundation Stereo](references/checkpoint-inference-mappings-foundation-stereo.md) <br>
- [Skill Info](references/skill_info.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive), 3 attempts per task, evaluated in k8s-sandbox environment. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Validates final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and expected workflow. <br>
- Efficiency: Tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code | Codex |
|---|---:|---:|
| Overall | 99.5% | 95.3% |
| Security | 100.0% | 100.0% |
| Correctness | 100.0% | 100.0% |
| Discoverability | 100.0% | 95.0% |
| Effectiveness | 100.0% | 83.3% |
| Efficiency | 97.3% | 98.3% |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
