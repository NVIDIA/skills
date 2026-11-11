## Description: <br>
Use when a concrete holoscan command in holoscan-applications or a standalone Module fails, hangs, regresses, or returns wrong output. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers working in a holoscan-applications checkout or a standalone Holoscan Module use this skill to turn one concrete failing, hanging, regressed, or semantically wrong Holoscan CLI command into a minimally fixed, reproducible passing command with focused regression proof. <br>

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
- [Holoscan CLI contract](references/holoscan-cli-contract.md) <br>
- [Holoscan CLI configuration](references/holoscan-cli-configuration.md) <br>
- [Holoscan CLI debugging workflow](references/debug-workflow.md) <br>
- [Version-sensitive diagnostic priors](references/known-issues.md) <br>
- [nvidia-holoscan/holoscan-applications](https://github.com/nvidia-holoscan/holoscan-applications) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Code] <br>
**Output Format:** [Markdown report with inline bash code blocks and code diffs] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Reports exact reproduction, environment and revision, primary layer, root cause, rejected hypotheses, minimal fix, passing proof, remaining uncertainty, and final worktree state; diagnostics are redacted before sharing] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (3 positive, 2 negative), 1 attempt per task, each run in an isolated sandbox pod; each task was also attempted without the skill as a baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` plus 50% `token_efficiency`. <br>

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
| Overall | 94.6% — baseline ran, but no comparable score was available; uplift unavailable | 76.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 60.0% → 76.0% (+16.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 56.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 57.0% → 78.0% (+21.0 points) | 58.0% → 61.8% (+3.8 points) |
| Efficiency | 94.8% — baseline ran, but no comparable score was available; uplift unavailable | 89.6% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
47e5238a (source: git SHA, committed 2026-10-20) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
