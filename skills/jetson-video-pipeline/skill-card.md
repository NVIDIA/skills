## Description: <br>
Execute and verify Jetson Video Codec SDK or PyNvVideoCodec encode/decode, transcode, segmentation, container decode, AV1, or acceptance workflows with exact artifact handoffs. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to execute official-sample codec stages on NVIDIA Jetson devices and prove that every consumer used the exact artifact produced by the preceding stage, including encode-then-decode verification, native transcode, PyNvVideoCodec segments, container decode triage, AV1 operation verification, and customer acceptance packages. <br>

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
- [Pipeline workflow](references/pipeline-workflow.md) <br>
- [Official sample contract](references/official-sample-contract.md) <br>
- [Agent Skills](https://agentskills.io/) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, JSON results, Artifact verification] <br>
**Output Format:** [JSON with structured pipeline result and workspace artifacts] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces provenance-tracked workspace with hashed artifacts and structured result JSON] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
2 evaluation tasks (2 positive) from skill-evaluator-dataset-snapshot/1, each run in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and expected workflow behavior was followed. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

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
| Overall | 50% → 100% (+50 points) | 35% → 71% (+36 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 40% → 100% (+60 points) | 40% → 30% (-10 points) |
| Discoverability | 50% → 100% (+50 points) | 22% → 94% (+72 points) |
| Effectiveness | 30% → 100% (+70 points) | 0% → 30% (+30 points) |
| Efficiency | 31% → 100% (+69 points) | 11% → 100% (+89 points) |

## Skill Version(s): <br>
e61c045 (source: git SHA, committed 2026-08-10) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
