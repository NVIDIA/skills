## Description: <br>
Used for finetuning NV-Generate-CTMR MR-Brain v1 for T1, T2, FLAIR, SWI, or MRA data from a NIfTI datalist. Not for clinical or production data approval. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers finetuning the NV-Generate-CTMR MR-Brain v1 diffusion UNet on custom T1, T2, FLAIR, SWI, or MRA NIfTI training volumes for research and development purposes. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NV-Generate-CTMR (upstream training scripts)](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR) <br>
- [NV-Generate-MR-Brain (HuggingFace model)](https://huggingface.co/nvidia/NV-Generate-MR-Brain) <br>
- [NV-Generate-CT (HuggingFace autoencoder)](https://huggingface.co/nvidia/NV-Generate-CT) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Shell commands, JSON] <br>
**Output Format:** [PyTorch checkpoint files, optional NIfTI inference images, and JSON result summary with provenance metadata] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Finetuned diffusion UNet checkpoint, optional inference outputs, and structured result JSON; all paths recorded in workflow_summary.json] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
2 evaluation tasks (2 positive) from skill-evaluator-dataset-snapshot/1, each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow (equal-weight mean of goal completion and behavior adherence). <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 48% → 98% (+50 points) | 65% → 96% (+32 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 40% → 100% (+60 points) | 100% → 100% (±0 points) |
| Discoverability | 41% → 100% (+59 points) | 41% → 91% (+50 points) |
| Effectiveness | 30% → 90% (+60 points) | 65% → 90% (+25 points) |
| Efficiency | 27% → 100% (+73 points) | 18% → 100% (+82 points) |

## Skill Version(s): <br>
0.1.0 (source: skill_manifest.yaml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
