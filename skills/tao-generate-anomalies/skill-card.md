## Description: <br>
Full PAIDF AnomalyGen pipeline — fine-tune on a new anomaly dataset, generate synthetic anomaly images (SDG), evaluate quality (nn_score), and search per-sample (guidance, crop_ratio) parameters across full, finetune_only, and inference_only modes. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who fine-tune PAIDF AnomalyGen on custom anomaly datasets, generate synthetic anomaly images, and evaluate and filter them by nn_score for visual inspection and defect-detection workflows. <br>

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
- [TAO Skill Bank Repository](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>
- [Agent Skills Open Standard](https://agentskills.io) <br>
- [Setup (HF_TOKEN, checkpoints, prerequisites)](references/setup.md) <br>
- [Fine-tuning guide](references/finetune.md) <br>
- [Fine-tuning commands](references/finetune-commands.md) <br>
- [Inference guide (Phases 2–7)](references/inference.md) <br>
- [Inference commands](references/inference-commands.md) <br>
- [Prep-testcase reference](references/prep-testcase.md) <br>
- [SDG inference reference](references/sdg-inference.md) <br>
- [SDG refine / search reference](references/sdg-refine.md) <br>
- [Evaluation scoring reference](references/eval.md) <br>
- [Guard preflight and custom counts](references/guard-and-custom-counts.md) <br>
- [Docker launch and permissions](references/docker.md) <br>
- [Datasets](references/datasets.md) <br>
- [Output layout and verification checklist](references/output-layout.md) <br>
- [Error handling](references/error-handling.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Generates synthetic anomaly images, per-sample CSVs, evaluation logs, and search/regen summary reports under results/<name>/] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 1 task (1 positive) in an isolated k8s-sandbox environment with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the security signal. <br>
- Correctness: Is the answer correct? Scored from the accuracy signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the skill_execution signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal_accuracy and behavior_check. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% skill_efficiency and 50% token_efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 99.4% — baseline ran, but no comparable score was available; uplift unavailable | 79.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 0.0% → 100.0% (+100.0 points) | 40.0% → 100.0% (+60.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 0.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 0.0% → 100.0% (+100.0 points) | 48.3% → 100.0% (+51.7 points) |
| Efficiency | 96.9% — baseline ran, but no comparable score was available; uplift unavailable | 99.7% → 98.4% (-1.3 points) |

## Skill Version(s): <br>
1.0.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
