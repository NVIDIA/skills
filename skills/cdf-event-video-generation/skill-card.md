## Description: <br>
Run the CDF Orchestration Event Video Generation DAG on Kubernetes or OSMO for image-to-video anomaly generation, auto-labeling, and anomaly dataset generation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to configure, trigger, monitor, and retrieve results from the CDF Orchestration Event Video Generation DAG on Kubernetes or OSMO. It generates image-to-video anomaly clips, auto-labels them, and builds anomaly datasets from seed images for safety and surveillance synthetic data generation. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Cloud Credentials, Other [cluster credential file (kubeconfig)]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [setup-and-preflight.md](references/setup-and-preflight.md) <br>
- [payload-contract.md](references/payload-contract.md) <br>
- [outputs.md](references/outputs.md) <br>
- [troubleshooting.md](references/troubleshooting.md) <br>
- [airflow-direct-api.md](references/airflow-direct-api.md) <br>
- [cdf-augmentation skill](https://github.com/NVIDIA/cdf-augmentation/tree/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation) <br>
- [cdf-auto-labeling skill](https://github.com/NVIDIA/cdf-auto-labeling/tree/6c75c76c52a6c76c3866a38d30885e11123b0981/skills/cdf-auto-labeling) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, API Calls, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks and rendered JSON payloads] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
15 evaluation tasks (15 positive), 1 attempt per task, each run in its own isolated k8s-sandbox pod (evaluator version 1.5.6, evaluation date 2026-10-21). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed, and were decoys avoided? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 88.7% (uplift unavailable) | 81.6% (uplift unavailable) |
| Security | 80.0% → 100.0% (+20.0 points) | 60.0% → 96.7% (+36.7 points) |
| Correctness | 36.0% → 97.3% (+61.3 points) | 30.7% → 82.7% (+52.0 points) |
| Discoverability | 77.5% (uplift unavailable) | 76.0% (uplift unavailable) |
| Effectiveness | 45.0% → 83.9% (+38.9 points) | 36.6% → 67.2% (+30.6 points) |
| Efficiency | 84.5% (uplift unavailable) | 85.5% (uplift unavailable) |

## Skill Version(s): <br>
1.0.2 (source: frontmatter metadata.version; repo pyproject.toml reports 1.1.0) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
