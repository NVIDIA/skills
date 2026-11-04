## Description: <br>
Run iterative DINOv3 SSL DEFT data selection and retraining from an unlabeled pool, using GRIT or multi-task weakness and fixed-encoder retrieval. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers use this skill to refine a DINOv3 representation by mining relevant samples from an unlabeled image pool (GRIT-score or multi-task round-robin) and retraining over iterative rounds inside a TAO Data Services container, with approval-gated launch, monitoring, and full data/model lineage. <br>

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
- [Customer Adapter Contracts](references/adapter-contracts.md) <br>
- [DINOv3 SSL DEFT Architecture](references/architecture.md) <br>
- [DINOv3 SSL DEFT container lifecycle](references/tao-container-integration.md) <br>
- [skill_info.yaml](references/skill_info.yaml) <br>
- [metrics.schema.json](references/metrics.schema.json) <br>
- [stage-request.schema.json](references/stage-request.schema.json) <br>
- [workflow.schema.json](references/workflow.schema.json) <br>
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks; YAML workflow configs; JSON/JSONL run state and Parquet manifests produced by the controller] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Launch requires explicit user approval of a six-item approval contract; monitoring continues until the workflow reaches a terminal state.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 positive evaluation tasks (approval monitoring, benchmark isolation, layered interface, traceability), 1 attempt per task, each in an isolated k8s sandbox pod, compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the security signal. <br>
- Correctness: Is the answer correct? Scored from final-answer accuracy. <br>
- Discoverability: Was the right skill loaded when needed? Scored from skill execution. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal accuracy and behavior check. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of tool productivity and token efficiency. <br>

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
| Overall | 96.1% — baseline ran, but no comparable score was available; uplift unavailable | 82.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 10.0% → 100.0% (+90.0 points) | 45.0% → 90.0% (+45.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 62.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 23.9% → 88.1% (+64.2 points) | 41.3% → 64.4% (+23.1 points) |
| Efficiency | 92.3% — baseline ran, but no comparable score was available; uplift unavailable | 96.3% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
