## Description: <br>
Compute federated statistics over tabular data (count, sum, mean, stddev, var, histogram, quantile, noise-protected min/max) and image data (count, failure_count, pixel-intensity histogram) across NVFLARE sites via FedStatsRecipe — automatic and non-interactive from the dataset, feature names (header or supplied), and optionally a README or notes declaring which statistics to compute. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers computing federated statistics across distributed NVFLARE sites for tabular and image data without writing custom aggregation code. <br>

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
- [Image Statistics Reference](references/image-statistics.md) <br>
- [Statistics Mapping Reference](references/statistics-mapping.md) <br>
- [Stats Job Validation Reference](references/stats-job-validation.md) <br>
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/en/main) <br>
- [NVIDIA FLARE Paper](https://arxiv.org/abs/2210.13291) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Analysis, Shell commands] <br>
**Output Format:** [Python source files (client.py, job.py) and JSON statistics output] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 17 positive evaluation tasks, each running in an isolated sandbox pod. Tier 3 live agent evaluation. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helps complete the user's goal and follows expected workflow. <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Codex (Baseline → Skill Uplift) |
|---|---:|
| Overall | 54% → 69% (+15 points) |
| Security | 88% → 47% (-41 points) |
| Correctness | 64% → 86% (+22 points) |
| Discoverability | 39% → 68% (+28 points) |
| Effectiveness | 41% → 71% (+30 points) |
| Efficiency | 38% → 72% (+34 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
