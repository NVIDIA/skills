## Description: <br>
Run TAO Data Services object-detection gap analysis from ground-truth and inference annotations, identifying weak images by comparing model predictions against ground truth using per-class recall, precision, and AP50 thresholds. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building NVIDIA TAO object-detection workflows use this skill to run Data Services gap analysis that compares inference annotations against ground truth (KITTI or COCO), computes per-image per-class TP/FP/FN/AP50 metrics, and produces a weak-image set for downstream data mining. <br>

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
- [skill_info.yaml](references/skill_info.yaml) <br>
- [default_object_detection.yaml (spec template)](assets/default_object_detection.yaml) <br>
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks; YAML spec; Parquet and JSON gap-analysis artifacts] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Writes box_gaps.parquet, image_metrics.parquet, weak_images.parquet, and gap_report.json to the configured results_dir] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive), 1 attempt per task, each run in an isolated k8s-sandbox pod; evaluator version 1.5.6, evaluated 2026-10-16. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the security signal. <br>
- Correctness: Is the answer correct? Scored from the accuracy signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the skill_execution signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal_accuracy and behavior_check. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of skill_efficiency and token_efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.1% — baseline ran, but no comparable score was available; uplift unavailable | 96.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 0.0% → 100.0% (+100.0 points) | 20.0% → 100.0% (+80.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 12.5% → 87.5% (+75.0 points) | 40.0% → 87.5% (+47.5 points) |
| Efficiency | 73.1% — baseline ran, but no comparable score was available; uplift unavailable | 98.5% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
