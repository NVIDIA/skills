## Description: <br>
How to swap the DeepStream CV detection model in the VSS Alerts Blueprint verification (2d_cv) mode — covers ONNX export, custom bbox parsers, compose mount gotchas, nvinfer config, runtime TRT engine build, deployment, and a segmentation-capable model addendum handoff. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to replace the stock CV detection model in the VSS Alerts Blueprint verification mode (2d_cv) with a custom ONNX-format detector, debug broken ONNX staging or parser load failures, and redeploy the perception-alerts service. <br>

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
- [Common Gotchas](references/common-gotchas.md) <br>
- [ds-start Entrypoint](references/ds-start-entrypoint.md) <br>
- [Segmentation Model Contract](references/segmentation-model-contract.md) <br>
- [VSS Source Layout](references/vss-source-layout.md) <br>
- [YOLOv11 ONNX Export](references/yolov11-onnx-export.md) <br>
- [YOLOv11 Parser](references/yolov11-parser.md) <br>
- [VSS Alerts Blueprint (GitHub)](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) <br>
- [VSS Quickstart](https://docs.nvidia.com/vss/latest/quickstart.html#download-the-deployment-package) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline bash code blocks and INI/YAML configuration snippets] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (3 positive) run in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helps complete the user's goal and follows the expected workflow (equal-weight mean of goal completion and behavior adherence). <br>
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
| Overall | 50% → 88% (+38 points) | Not available |
| Security | 100% → 67% (-33 points) | Not available |
| Correctness | 67% → 100% (+33 points) | Not available |
| Discoverability | 17% → 98% (+81 points) | Not available |
| Effectiveness | 52% → 82% (+31 points) | Not available |
| Efficiency | 14% → 93% (+79 points) | Not available |

## Skill Version(s): <br>
7ecae6b (source: git SHA, committed 2026-07-31) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
