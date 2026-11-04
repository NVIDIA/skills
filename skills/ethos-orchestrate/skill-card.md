## Description: <br>
Entry point for an AI agent's ETHOS.md (a durable snapshot of what the agent is for, what it must not do, and what success looks like) that routes requests to explore and capture intent (ethos-explore) or to write or edit the file (ethos-write). <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers building AI agents use this skill with a coding assistant to find, explain, create or update their agent's ETHOS.md, routing each request to the ethos-explore or ethos-write sub-flow while ensuring the file records only user-confirmed intent. Interrupted conversations can resume from supported prior context without repeating completed questions or assuming unconfirmed approval. <br>

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
- [ETHOS.md overview (README.md)](README.md) <br>
- [ethos-explore skill](../ethos-explore/SKILL.md) <br>
- [ethos-write skill](../ethos-write/SKILL.md) <br>
- [ETHOS.md schema-v1 template](../ethos-write/references/templates/ETHOS.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Files] <br>
**Output Format:** [Markdown with inline bash code blocks; routed sub-flows produce a single ETHOS.md file] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Routing reads the local checkout only and works offline; writes are limited to the ETHOS.md file via sub-flows and nothing is committed automatically.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), 1 attempt per task, each run in an isolated sandbox pod (`k8s-sandbox`) and compared against a no-skill baseline; evaluator version 1.5.6, evaluated 2026-10-23. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal (100%). <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal (100%). <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` plus 50% `token_efficiency`. <br>

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
| Overall | 92.2% (baseline not comparable; uplift unavailable) | 93.7% (baseline not comparable; uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 65.0% → 100.0% (+35.0 points) | 85.0% → 90.0% (+5.0 points) |
| Discoverability | 76.7% (baseline not comparable; uplift unavailable) | 93.3% (baseline not comparable; uplift unavailable) |
| Effectiveness | 56.0% → 93.1% (+37.1 points) | 64.8% → 89.2% (+24.4 points) |
| Efficiency | 91.0% (baseline not comparable; uplift unavailable) | 96.0% (baseline not comparable; uplift unavailable) |

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml; git SHA 97cd9ee, committed 2026-10-23) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
