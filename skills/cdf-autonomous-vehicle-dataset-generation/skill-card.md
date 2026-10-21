## Description: <br>
Prepare payloads, run and monitor the Autonomous Vehicle Dataset Generation Airflow workflow, and inspect results for driving-clip retrieval and Cosmos3-Nano variants. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to prepare payloads for, submit, monitor, and inspect results of the Cosmos Data Factory Autonomous Vehicle Dataset Generation workflow, retrieving driving clips similar to an example video or generating rain, snow, and daylight variations with Cosmos3-Nano. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Other [Basic auth], Cloud Credentials] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Direct Airflow API](references/airflow-direct-api.md) <br>
- [Autonomous Vehicle Dataset Generation Payload Guide](../../docs/autonomous-vehicle-dataset-generation/payload-guide.md) <br>
- [Autonomous Vehicle Dataset Generation Getting Started](../../docs/autonomous-vehicle-dataset-generation/getting-started.md) <br>
- [Autonomous Vehicle Dataset Generation Prerequisites](../../docs/autonomous-vehicle-dataset-generation/prerequisites.md) <br>
- [Use Azure Storage with AV Dataset Generation](../../docs/autonomous-vehicle-dataset-generation/azure-support.md) <br>
- [cdf-curation-and-retrieval skill](https://github.com/NVIDIA/cdf-curation-and-retrieval/tree/b4d553d00cc00c1293ad997fa8693af809489c71/skills/cdf-curation-and-retrieval) <br>
- [cdf-augmentation skill](https://github.com/NVIDIA/cdf-augmentation/tree/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Files, Shell commands, API Calls, Analysis] <br>
**Output Format:** [Markdown with JSON payloads and inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative), 1 attempt per task, each run in an isolated sandbox pod in a `k8s-sandbox` environment; each task was also attempted without the skill as a baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of `skill_efficiency` and `token_efficiency`. <br>

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
| Overall | 94.6% — baseline ran, but no comparable score was available; uplift unavailable | 87.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 66.7% → 100.0% (+33.3 points) | 100.0% → 83.3% (-16.7 points) |
| Correctness | 23.3% → 100.0% (+76.7 points) | 23.3% → 100.0% (+76.7 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 19.5% → 88.1% (+68.6 points) | 16.3% → 73.1% (+56.8 points) |
| Efficiency | 85.1% — baseline ran, but no comparable score was available; uplift unavailable | 91.2% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.1.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
