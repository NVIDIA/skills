## Description: <br>
Query Auto Ontology and validate generated SQL, rows, and answers, using MCP or REST access, readiness checks, authentication, conversations, and grounded questions. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and partners embedding Auto Ontology in an agent harness use this skill to check readiness, discover semantic-layer terms, ask grounded descriptive or diagnostic questions through MCP or the authenticated REST gateway, and validate the generated SQL, returned rows, truncation, and answer prose before presenting a claim. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, OAuth Token] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Grounded query and answer validation](references/query-validation.md) <br>
- [Auto Ontology in a broader stack](references/stack.md) <br>
- [Runtime contract](assets/runtime-contract.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [API Calls, Analysis] <br>
**Output Format:** [Markdown with MCP tool-call sequences and an inline Python requests example against the Next.js /api gateway] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Read-only; ask_question results are capped at 100 rows with row_count and truncated flags; unverifiable material constraints are reported as gaps rather than stronger claims] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 positive evaluation tasks, one attempt per task, each run in an isolated k8s sandbox pod and compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? (scored from `security`) <br>
- Correctness: Is the answer correct? (scored from `accuracy`) <br>
- Discoverability: Was the right skill loaded when needed? (scored from `skill_execution`) <br>
- Effectiveness: Did the skill help complete the task? (50% `goal_accuracy` + 50% `behavior_check`) <br>
- Efficiency: Did it avoid wasted tool calls and token usage? (50% `skill_efficiency` + 50% `token_efficiency`) <br>

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
| Overall | 95.3% — baseline ran, but no comparable score was available; uplift unavailable | 80.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 84.6% → 92.3% (+7.7 points) |
| Correctness | 47.7% → 100.0% (+52.3 points) | 50.8% → 70.8% (+20.0 points) |
| Discoverability | 84.6% — baseline ran, but no comparable score was available; uplift unavailable | 75.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 55.5% → 96.0% (+40.5 points) | 48.8% → 68.5% (+19.7 points) |
| Efficiency | 95.6% — baseline ran, but no comparable score was available; uplift unavailable | 95.2% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.2.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
