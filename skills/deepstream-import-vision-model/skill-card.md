## Description: <br>
Use this skill to bring a supported object-detection vision model from HuggingFace or NVIDIA NGC into an NVIDIA DeepStream pipeline with end-to-end automation: ONNX download, SafeTensors export, TRT engine build, custom nvinfer bbox parser, multi-stream benchmark, and PDF report. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to import supported object-detection vision models from HuggingFace or NVIDIA NGC into NVIDIA DeepStream inference pipelines, automating the full workflow from model acquisition through TensorRT engine build, multi-stream benchmarking, and PDF report generation. <br>

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
- [Model Acquire](references/model-acquire.md) <br>
- [Engine Build](references/engine-build.md) <br>
- [Pipeline Run](references/pipeline-run.md) <br>
- [Report Generation](references/report-generation.md) <br>
- [Windows Support](references/windows.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Code, Files, Analysis] <br>
**Output Format:** [TensorRT engines, ONNX models, C++ parser source, nvinfer configs, benchmark logs, PDF/HTML/Markdown reports] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Outputs organized in a mandatory directory structure under models/{model_name}/] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 evaluation tasks (13 positive), each run in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use — detects unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the final answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal (goal completion and expected workflow adherence). <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `skill_efficiency`: Evaluates routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Measures final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Measures whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 44% → 72% (+27 points) | 46% → 61% (+15 points) |
| Security | 58% → 54% (-4 points) | 35% → 42% (+8 points) |
| Correctness | 42% → 86% (+45 points) | 65% → 72% (+8 points) |
| Discoverability | 51% → 88% (+37 points) | 50% → 72% (+22 points) |
| Effectiveness | 24% → 43% (+19 points) | 28% → 34% (+6 points) |
| Efficiency | 48% → 86% (+38 points) | 54% → 84% (+30 points) |

## Skill Version(s): <br>
1.5.2 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
