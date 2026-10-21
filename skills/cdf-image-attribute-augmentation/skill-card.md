## Description: <br>
Run the CDF Orchestration Image Attribute Augmentation DAG on Kubernetes - person-crop clothing augmentation, attribute search, and augmented dataset generation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to configure, trigger, and monitor the Image Attribute Augmentation DAG on a CDF Orchestration Airflow/Kubernetes deployment, generating augmented person-attribute datasets (clothing variations, attribute search, and captions) and retrieving the results. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Cloud Credentials, Other [cluster credential file]] <br>

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
**Output Type(s):** [Shell commands, API Calls, Analysis, Files] <br>
**Output Format:** [Markdown with inline bash code blocks and rendered JSON payloads] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 evaluation tasks (11 positive, 2 negative), 1 attempt per task, each run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was loaded and activated when needed, avoiding decoys. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 84.1% (uplift unavailable) | 82.0% (uplift unavailable) |
| Security | 84.6% → 100.0% (+15.4 points) | 88.5% → 100.0% (+11.5 points) |
| Correctness | 29.2% → 86.2% (+57.0 points) | 30.8% → 76.9% (+46.1 points) |
| Discoverability | 79.6% (uplift unavailable) | 73.2% (uplift unavailable) |
| Effectiveness | 37.6% → 67.5% (+29.9 points) | 34.3% → 67.1% (+32.8 points) |
| Efficiency | 87.4% (uplift unavailable) | 92.8% (uplift unavailable) |

## Skill Version(s): <br>
1.0.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
