## Description: <br>
Run iterative improvement for NVIDIA TAO CLIP / SigLIP image-text retrieval on attribute-labelled data, performing dataset preparation, zero-shot evaluation, attribute gap analysis, caption-space k-NN mining, history-aware selection, retraining, and re-evaluation until a retrieval KPI, validation plateau, or iteration budget stops the loop. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers use this skill to orchestrate iterative DEFT (Dataset Evolution through Fine-Tuning) workflows for NVIDIA TAO CLIP/SigLIP image-text retrieval models on People Attribute Search (PAS) attribute-labelled datasets, running a bounded, resumable evaluate-mine-retrain loop on Docker, SLURM, Kubernetes, Brev, or virtualenv execution platforms. <br>

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
**Output Type(s):** [Shell commands, Configuration instructions, Analysis, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces an HTML loop report, CSV metrics, Parquet datasets, run state/log JSON, and checkpoint artifacts under the results directory] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
20 evaluation tasks (20 positive), 1 attempt per task, each in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded when needed. <br>
- Effectiveness: Checks goal completion and expected workflow behavior adherence (equal weight). <br>
- Efficiency: Checks tool-call productivity and token usage efficiency (equal weight). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 89.4% | 69.7% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 35.0% → 95.0% (+60.0 points) | 45.0% → 79.0% (+34.0 points) |
| Discoverability | 67.0% | 9.5% |
| Effectiveness | 30.0% → 85.5% (+55.5 points) | 29.4% → 59.8% (+30.4 points) |
| Efficiency | 99.5% | 100.0% |

## Skill Version(s): <br>
0.4.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
