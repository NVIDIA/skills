## Description: <br>
Used for generating synthetic T1, T2, FLAIR, SWI, or MRA brain MRI volumes with NV-Generate-CTMR MR-Brain v1. Not for production training data. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and researchers use this skill to generate synthetic brain MRI volumes across T1, T2, FLAIR, SWI, and MRA modalities for research, development, and demonstration purposes. <br>

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
- [FOV And Downloads](references/fov-and-downloads.md) <br>
- [NVIDIA-Medtech/NV-Generate-CTMR (upstream, pinned commit)](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/tree/da438fec6484cdb6f421f8c7051d954ebefff730) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Analysis] <br>
**Output Format:** [NIfTI volumes with JSON summary] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Output volumes are synthetic; not safe as production training data without independent quality review] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (3 positive), each attempt in an isolated sandbox pod. Tier 3 live agent evaluation. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Checks whether the answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow. <br>
- Efficiency: Checks whether the skill avoided wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 45% → 98% (+53 points) | 58% → 99% (+41 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 20% → 100% (+80 points) | 53% → 100% (+47 points) |
| Discoverability | 50% → 90% (+40 points) | 56% → 94% (+38 points) |
| Effectiveness | 18% → 100% (+82 points) | 51% → 100% (+49 points) |
| Efficiency | 39% → 100% (+61 points) | 27% → 100% (+73 points) |

## Skill Version(s): <br>
78b8944 (source: git SHA, committed 2026-08-29) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
