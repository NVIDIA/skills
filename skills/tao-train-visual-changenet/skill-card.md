## Description: <br>
Visual ChangeNet for binary image classification and segmentation in AOI defect detection. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers training, evaluating, exporting, and running inference on Visual ChangeNet models for automated optical inspection (AOI) defect detection, PCB visual inspection, and change-segmentation workflows. <br>

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
- [data-formats.md](references/data-formats.md) <br>
- [dinov3-backbones.md](references/dinov3-backbones.md) <br>
- [local-docker.md](references/local-docker.md) <br>
- [parent-model-inference.md](references/parent-model-inference.md) <br>
- [tuning-parameters.md](references/tuning-parameters.md) <br>
- [troubleshooting.md](references/troubleshooting.md) <br>
- [tao-deploy-visual-changenet.md](references/tao-deploy-visual-changenet.md) <br>
- [skill_info.yaml](references/skill_info.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline bash and Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive), 3 attempts per task, in k8s-sandbox environment. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Whether the user's goal was achieved and expected workflow behavior was followed. <br>
- Efficiency: Tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 99.3% | 72.4% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 6.7% → 100.0% (+93.3 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% | 0.0% |
| Effectiveness | 12.2% → 100.0% (+87.8 points) | 70.0% → 63.3% (-6.7 points) |
| Efficiency | 96.5% | 99.4% → 98.4% (-1.0 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
