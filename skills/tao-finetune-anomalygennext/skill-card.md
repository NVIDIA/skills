## Description: <br>
Validate a user AnomalyGenNext dataset and recipe, freeze its validation set, and prepare a canonical texture fine-tuning recipe for task-specific AnomalyGenNext LoRA training when task weights do not yet exist. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to validate a user-owned AnomalyGenNext defect dataset and validation set, prepare a canonical fine-tuning recipe, and launch and accept task-specific AnomalyGenNext LoRA training in the pinned AnomalyGenNext 1.1 container, accepting results only when the NN score strictly improves over the iteration-zero baseline. <br>

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
- [Container runtime](references/container-runtime.md) <br>
- [Fine-tuning input contract](references/input-contract.md) <br>
- [NN-score completion contract](references/nn-score-contract.md) <br>
- [skill_info.yaml](references/skill_info.yaml) <br>
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks; canonical recipe YAML, normalized validation JSONL, metadata, and training_handoff.json files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Training success is accepted only when training_handoff.json is COMPLETE, with a hash-bound adapter checkpoint and recipe and a strictly improved Average.nn_score over the iteration-zero baseline.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive: prepare-anomalygennext-finetune-recipe), 1 attempt per task, each run in an isolated k8s-sandbox pod, compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the security signal. <br>
- Correctness: Is the answer correct? Scored from final-answer accuracy. <br>
- Discoverability: Was the right skill loaded when needed? Scored from skill execution. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal accuracy and behavior check. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of tool-call productivity and token efficiency. <br>

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
| Overall | 93.2% — baseline ran, but no comparable score was available; uplift unavailable | 57.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 0.0% → 100.0% (+100.0 points) | 80.0% → 100.0% (+20.0 points) |
| Discoverability | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 20.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 0.0% → 75.0% (+75.0 points) | 27.5% → 27.5% (±0.0 points) |
| Efficiency | 96.0% — baseline ran, but no comparable score was available; uplift unavailable | 41.1% — baseline ran, but no comparable score was available; uplift unavailable |

Overall verdict: PASS — Recommended for publication.

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
