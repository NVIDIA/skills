## Description: <br>
Used for running NV-Segment-CTMR on CT or MRI NIfTI volumes and recording label-map evidence. Not for clinical interpretation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run CT or MRI NIfTI volume segmentation using NVIDIA's NV-Segment-CTMR MONAI bundle and record structured label-map evidence for engineering verification. <br>

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
- [NVIDIA-Medtech/NV-Segment-CTMR upstream bundle](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/tree/cb921f5c58837c0f42a713855d68b32af88e1cdd/NV-Segment-CTMR) <br>
- [BENCHMARK.md](BENCHMARK.md) <br>


## Skill Output: <br>
**Output Type(s):** [JSON, Files] <br>
**Output Format:** [JSON evidence file with NIfTI label-map output] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
2 evaluation tasks (2 positive) from a versioned dataset (digest sha256:d9381987…). Each task attempt ran in its own isolated sandbox pod with 3 attempts per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and activated when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal accuracy + 50% behavior check). <br>
- Efficiency: Whether tool calls and token usage were productive (50% tool productivity + 50% token efficiency). <br>

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
| Overall | 88.7% | 79.7% |
| Security | 100.0% → 100.0% (±0.0 pts) | 100.0% → 50.0% (-50.0 pts) |
| Correctness | 30.0% → 100.0% (+70.0 pts) | 80.0% → 100.0% (+20.0 pts) |
| Discoverability | 95.0% | 85.0% |
| Effectiveness | 26.0% → 66.7% (+40.7 pts) | 27.1% → 95.0% (+67.9 pts) |
| Efficiency | 81.6% | 68.7% |

## Skill Version(s): <br>
ef18b91 (source: git SHA, committed 2026-09-14) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
