## Description: <br>
Run iterative improvement for NVIDIA TAO CLIP / SigLIP image-text retrieval on attribute-labelled data. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers use this skill to orchestrate iterative DEFT (Dataset Evolution through Fine-Tuning) workflows for NVIDIA TAO CLIP/SigLIP image-text retrieval models on People Attribute Search (PAS) attribute-labelled datasets, including dataset preparation, zero-shot evaluation, attribute gap analysis, caption-space k-NN mining, retraining, and re-evaluation. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NVIDIA TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>
- [Agent Skills Open Standard](https://agentskills.io) <br>
- [Preflight and Initialization](references/preflight.md) <br>
- [Pipeline and State](references/pipeline-and-state.md) <br>
- [Scripts and Agents](references/scripts-and-agents.md) <br>
- [Platform Execution](references/platform-execution.md) <br>
- [Data Layout](references/data-layout.md) <br>
- [Metric Contract](references/metric-contract.md) <br>
- [Mining](references/mining.md) <br>
- [CLIP Train and Evaluate](references/clip-train-eval.md) <br>
- [Gap Analysis](references/gap-analysis.md) <br>
- [Visualization](references/visualization.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces HTML reports, CSV metrics, Parquet datasets, and checkpoint artifacts under the results directory] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
14 evaluation tasks (14 positive), 3 attempts per task, in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Checks goal completion and expected workflow behavior adherence. <br>
- Efficiency: Checks tool-call productivity and token usage efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Checks final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the user's goal was achieved. <br>
- `behavior_check`: Checks whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Checks tool-call productivity. <br>
- `token_efficiency`: Checks actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 85.7% | 62.5% |
| Security | 100.0% → 100.0% (±0.0 pts) | 100.0% → 100.0% (±0.0 pts) |
| Correctness | 21.3% → 96.0% (+74.7 pts) | 33.3% → 67.8% (+34.5 pts) |
| Discoverability | 54.7% | 0.0% |
| Effectiveness | 25.1% → 80.2% (+55.1 pts) | 25.0% → 45.4% (+20.4 pts) |
| Efficiency | 97.8% | 99.6% → 99.2% (-0.4 pts) |

## Skill Version(s): <br>
0.4.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
