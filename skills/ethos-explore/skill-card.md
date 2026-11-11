## Description: <br>
Captures what an AI agent is supposed to do — its purpose, boundaries, and success criteria — by exploring the user's codebase and docs first, then asking one intent question at a time for what source cannot supply, and handing the answers to ethos-write, which writes ethos.md. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building, evaluating, reviewing, or changing an AI agent use this skill to scan the agent's repository and run a short intent interview that captures the agent's role, purpose, scope, principles, constraints, and vision as a draft ethos.md for ethos-write. <br>

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


## Skill Output: <br>
**Output Type(s):** [Analysis, Interview questions, Draft ethos.md content] <br>
**Output Format:** [Markdown (one intent question per message, then a full draft ethos.md with one ## section per field)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Read-only codebase pre-flight; writes no files itself and hands the structured conversation to ethos-write.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), 1 attempt per task, each run in an isolated sandbox pod and compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? <br>
- Correctness: Is the answer correct? <br>
- Discoverability: Was the right skill loaded when needed? <br>
- Effectiveness: Did the skill help complete the task? (equal-weight goal completion and expected workflow adherence) <br>
- Efficiency: Did it avoid wasted tool calls and token usage? (equal-weight tool-call productivity and token efficiency) <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.3% (uplift unavailable) | 88.2% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 65.0% → 90.0% (+25.0 points) | 70.0% → 85.0% (+15.0 points) |
| Discoverability | 100.0% (uplift unavailable) | 90.0% (uplift unavailable) |
| Effectiveness | 55.4% → 75.0% (+19.6 points) | 46.7% → 68.1% (+21.4 points) |
| Efficiency | 91.4% (uplift unavailable) | 98.0% (uplift unavailable) |

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
