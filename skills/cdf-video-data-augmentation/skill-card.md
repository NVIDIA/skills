## Description: <br>
Run the CDF Video Data Augmentation Airflow workflow on Kubernetes or OSMO to create Cosmos video-to-video variants and auto-label both source and generated videos. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to author and validate a VDA trigger payload, preflight an existing Airflow/Kubernetes or OSMO installation, submit and monitor one video-data-augmentation run, and verify and report its Cosmos variant and auto-labeling outputs. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Cloud Credentials] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [setup-and-preflight.md](references/setup-and-preflight.md) <br>
- [payload-contract.md](references/payload-contract.md) <br>
- [airflow-direct-api.md](references/airflow-direct-api.md) <br>
- [outputs.md](references/outputs.md) <br>
- [troubleshooting.md](references/troubleshooting.md) <br>
- [Video Data Augmentation Getting Started](../../docs/video-data-augmentation/getting-started.md) <br>
- [Video Data Augmentation Payload Guide](../../docs/video-data-augmentation/payload-guide.md) <br>
- [cdf-augmentation skill](https://github.com/NVIDIA/cdf-augmentation/tree/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation) <br>
- [cdf-auto-labeling skill](https://github.com/NVIDIA/cdf-auto-labeling/tree/6c75c76c52a6c76c3866a38d30885e11123b0981/skills/cdf-auto-labeling) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Shell commands, Analysis] <br>
**Output Format:** [Markdown with JSON payloads and inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
21 evaluation tasks (16 positive, 5 negative), one attempt per task, each running in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Checks final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? Checks whether the expected skill was selected and decoys avoided. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Dimension | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.1% (uplift unavailable) | 86.9% (uplift unavailable) |
| Security | 95.2% → 100.0% (+4.8 points) | 83.3% → 90.5% (+7.2 points) |
| Correctness | 36.2% → 90.5% (+54.3 points) | 49.5% → 82.9% (+33.4 points) |
| Discoverability | 89.4% (uplift unavailable) | 88.1% (uplift unavailable) |
| Effectiveness | 35.6% → 87.6% (+52.0 points) | 39.96% → 79.04% (+39.08 points) |
| Efficiency | 93.0% (uplift unavailable) | 93.9% (uplift unavailable) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter; repo pyproject.toml reports 1.1.0) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
