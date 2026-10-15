## Description: <br>
Entry point for an AI agent's ethos.md (a durable snapshot of what the agent is for, what it must not do, and what success looks like) that routes a request to explore and capture intent (ethos-explore) or to write or edit the file (ethos-write), including renaming an Eval Author 0.1.0 ETHOS.md, and owns the standard every ethos sub-flow follows. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers building AI agents use this skill to create, review, update, or rename their agent's ethos.md; it locates any existing file, picks the right ethos sub-flow, and makes sure the file records intent the user confirmed rather than intent inferred from code. <br>

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
- [ethos-explore sub-flow](../ethos-explore/SKILL.md) <br>
- [ethos-write sub-flow](../ethos-write/SKILL.md) <br>
- [ethos.md template](../ethos-write/references/templates/ethos.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Files] <br>
**Output Format:** [Markdown with inline bash code blocks; routes to sub-flows that write a local ethos.md] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Routing reads the local checkout only and works offline; nothing is committed automatically] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), 1 attempt per task, each run in an isolated sandbox pod (`k8s-sandbox`) with SkillEvaluator 1.5.6; every task was also run without the skill as a baseline. <br>

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
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.3% (no comparable baseline; uplift unavailable) | 96.8% (no comparable baseline; uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 50.0% → 90.0% (+40.0 points) | 65.0% → 100.0% (+35.0 points) |
| Discoverability | 80.0% (no comparable baseline; uplift unavailable) | 91.7% (no comparable baseline; uplift unavailable) |
| Effectiveness | 53.8% → 95.0% (+41.2 points) | 53.5% → 94.2% (+40.7 points) |
| Efficiency | 96.6% (no comparable baseline; uplift unavailable) | 98.3% (no comparable baseline; uplift unavailable) |

Overall verdict: PASS. Tier 1 static validation passed with observations (11 validators, 7 findings); Tier 2 semantic deduplication passed; Tier 3 live agent evaluation passed (2 agents, 4 tasks). Across all agents, the skill used 38.59% fewer tokens than the baseline (1,486,576 vs. 2,420,740).

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml; git SHA ba3626f, committed 2026-10-15) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
