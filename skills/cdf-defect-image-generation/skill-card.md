## Description: <br>
Generate synthetic PCB inspection images on Kubernetes with the SMT Defect Image Generation (DIG) DAG in CDF Orchestration, which renders components in simulation, turns the renders into realistic golden images (sim2real), and uses AnomalyGen to add texture defects such as solder bridges, excess solder or missing components. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building surface-mount technology (SMT) PCB inspection datasets use this skill to build and check DIG payloads, pick a pipeline mode (full, simulation, sim2real or defect-only), check cluster and GPU readiness, trigger and follow a run, and match each defect image to its golden and simulated source. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Other [Basic auth], Other [Kubernetes cluster credential file]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Payload contract](references/payload-contract.md) <br>
- [Setup and preflight](references/setup-and-preflight.md) <br>
- [Airflow direct API](references/airflow-direct-api.md) <br>
- [Outputs](references/outputs.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [cdf-simulation skill (pinned)](https://github.com/NVIDIA/cdf-simulation/tree/a0af274e6352347a1979be40036efdd33b447272/skills/cdf-simulation) <br>
- [Simulation YAML JobConfig reference](https://github.com/NVIDIA/cdf-simulation/blob/a0af274e6352347a1979be40036efdd33b447272/pcb_aoi/config/README.md) <br>
- [cdf-augmentation skill (pinned)](https://github.com/NVIDIA/cdf-augmentation/tree/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation) <br>
- [Augmentation configuration schema](https://github.com/NVIDIA/cdf-augmentation/blob/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation/references/configuration-schema.md) <br>
- [Augmentation captioning strategy guide](https://github.com/NVIDIA/cdf-augmentation/blob/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation/references/captioning-strategy-guide.md) <br>
- [cdf-anomalygen skill (pinned)](https://github.com/NVIDIA/cdf-anomalygen/tree/63c888cb479e22f6abe343421eaba2ae1847b8f5/skills/cdf-anomalygen) <br>
- [AnomalyGen inputs](https://github.com/NVIDIA/cdf-anomalygen/blob/63c888cb479e22f6abe343421eaba2ae1847b8f5/skills/cdf-anomalygen/references/inputs.md) <br>
- [AnomalyGen agentic mask generation](https://github.com/NVIDIA/cdf-anomalygen/blob/63c888cb479e22f6abe343421eaba2ae1847b8f5/skills/cdf-anomalygen/references/agentic-mask-generation.md) <br>
- [AnomalyGen run_pipeline config reference](https://github.com/NVIDIA/cdf-anomalygen/blob/63c888cb479e22f6abe343421eaba2ae1847b8f5/docs/pipeline/config.md) <br>


## Skill Output: <br>
**Output Type(s):** [Files, API Calls, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline JSON payloads and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Payloads are checked locally with the DAG's own model and compiler before any submission; execution requires explicit user confirmation and submits exactly one run.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
18 evaluation tasks (12 positive, 6 negative), one attempt per task, each run in an isolated k8s-sandbox pod; Tier 1 static validation, Tier 2 semantic deduplication and Tier 3 live agent evaluation. Overall verdict: PASS. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks unsafe operations, secret leakage and unauthorized access. <br>
- Correctness: Is the answer correct? Final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% tool-call productivity and 50% token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 93.3% (uplift unavailable) | 92.9% (uplift unavailable) |
| Security | 94.4% → 100.0% (+5.6 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 43.3% → 86.7% (+43.4 points) | 61.1% → 87.8% (+26.7 points) |
| Discoverability | 100.0% (uplift unavailable) | 91.3% (uplift unavailable) |
| Effectiveness | 42.1% → 85.8% (+43.7 points) | 48.8% → 88.3% (+39.5 points) |
| Efficiency | 93.8% (uplift unavailable) | 97.4% (uplift unavailable) |

## Skill Version(s): <br>
1.0.4 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
