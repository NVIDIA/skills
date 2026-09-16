## Description: <br>
Use when planning, executing, and independently validating Jetson Video Codec SDK or PyNvVideoCodec encode/decode, transcode, segmentation, container decode, AV1, or concise acceptance workflows. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to plan, execute, and validate Jetson Video Codec SDK and PyNvVideoCodec codec workflows including encode/decode, transcode, segmentation, container decode, and AV1 verification on NVIDIA Jetson devices. <br>

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
- [Pipeline Workflow](references/pipeline-workflow.md) <br>
- [Official Sample Contract](references/official-sample-contract.md) <br>
- [Buffer Sharing and Synchronization](references/buffer-sharing-and-synchronization.md) <br>
- [In-Process Codec Boundaries](references/in-process-codec-boundaries.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline JSON evidence blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 internal evaluation tasks (4 positive) with 3 attempts per task in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and the expected workflow behavior was followed. <br>
- Efficiency: Checks tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies expected skill selection, decoy avoidance, and workflow execution. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Measures tool-call productivity. <br>
- `token_efficiency`: Measures actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 88.7% — uplift unavailable | 93.2% — uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 66.7% → 100.0% (+33.3 points) |
| Correctness | 100.0% → 85.0% (-15.0 points) | 66.7% → 100.0% (+33.3 points) |
| Discoverability | 95.0% — uplift unavailable | 93.8% — uplift unavailable |
| Effectiveness | 55.4% → 67.5% (+12.1 points) | 34.2% → 88.8% (+54.6 points) |
| Efficiency | 96.0% — uplift unavailable | 83.4% — uplift unavailable |

## Skill Version(s): <br>
74bce4d (source: git SHA, committed 2026-09-16) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
