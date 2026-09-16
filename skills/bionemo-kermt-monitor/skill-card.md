## Description: <br>
Check progress for a detached KERMT run (pretrain, finetune, or any kermt_run_detached invocation). Reads run.json, queries docker for container state, tails the pretrain/finetune log, and parses progress lines (epoch, step, val loss). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers monitoring detached KERMT training and finetuning runs to check container state, training progress, and metrics without interrupting the running job. <br>

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
- [KERMT: Multitask finetuning and acceleration of chemical pretrained models](https://arxiv.org/abs/2510.12719) <br>
- [GROVER: Self-Supervised Message Passing Transformer on Large-Scale Molecular Data](https://arxiv.org/abs/2007.02835) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Supports --json flag for structured JSON output] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), each with 3 attempts per task in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Checks final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? Checks whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% tool-call productivity and 50% token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Checks final-answer correctness against the reference answer. <br>
- `skill_execution`: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Checks whether the user's goal was achieved. <br>
- `behavior_check`: Checks whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Measures tool-call productivity. <br>
- `token_efficiency`: Measures actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.6% | 76.5% |
| Security | 100.0% → 100.0% (±0.0 points) | 50.0% → 75.0% (+25.0 points) |
| Correctness | 28.0% → 100.0% (+72.0 points) | 32.0% → 70.0% (+38.0 points) |
| Discoverability | 96.7% | 91.7% |
| Effectiveness | 31.5% → 79.4% (+47.9 points) | 27.5% → 51.3% (+23.8 points) |
| Efficiency | 86.8% | 94.8% |

## Skill Version(s): <br>
77111e0 (source: git SHA, committed 2026-09-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
