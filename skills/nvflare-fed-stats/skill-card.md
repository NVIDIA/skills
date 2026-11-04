## Description: <br>
Compute federated statistics over tabular data (count, sum, mean, stddev, var, histogram, quantile, noise-protected min/max) and image data (count, failure_count, pixel-intensity histogram) across NVFLARE sites via FedStatsRecipe — automatic and non-interactive from the dataset, feature names (header or supplied), and optionally a README or notes declaring which statistics to compute. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers computing federated statistics (counts, means, histograms, quantiles) across distributed NVFLARE sites for tabular or image datasets, enabling privacy-preserving data summaries without exposing raw data. <br>

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
- [image-statistics.md](references/image-statistics.md) <br>
- [statistics-mapping.md](references/statistics-mapping.md) <br>
- [stats-job-validation.md](references/stats-job-validation.md) <br>
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/en/main) <br>
- [NVIDIA FLARE Paper](https://arxiv.org/abs/2210.13291) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Shell commands] <br>
**Output Format:** [JSON statistics output with generated Python files (client.py, job.py)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Privacy filters applied by default; aggregates only, never raw data] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
17 evaluation tasks (17 positive) executed in isolated k8s-sandbox pods, 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Goal completion (50%) and expected workflow adherence (50%). <br>
- Efficiency: Tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 70.6% |
| Security | Not available | 76.5% → 47.1% (-29.4 points) |
| Correctness | Not available | 67.1% → 85.9% (+18.8 points) |
| Discoverability | Not available | 72.7% |
| Effectiveness | Not available | 50.5% → 69.8% (+19.3 points) |
| Efficiency | Not available | 77.7% |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
