## Description: <br>
Diagnose a single failed, crashed, stopped, OOMKilled, CrashLoopBackOff, or ImagePullBackOff Run:ai workload by classifying its failure, citing quantified evidence, identifying recurrence, and recommending the next action. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers, ML engineers, and platform operators running NVIDIA Run:ai clusters use this skill to investigate why a single workload failed, read-only, and receive a classified root cause with quantified evidence and a recommended next action. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Service Account] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [failure-taxonomy.md](references/failure-taxonomy.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis] <br>
**Output Format:** [Markdown diagnostic report with a leading failure category, cited evidence, recurrence count, and recommendations] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Read-only: the skill inspects and reports only, recommending changes but never modifying workloads or cluster resources; credentials and sensitive values are redacted when citing evidence.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
22 evaluation tasks (19 positive, 3 negative) run with and without the skill across two agents, each attempt in an isolated k8s-sandbox pod; dataset digest sha256:11671384f2bddc73199555ab2ec1df1138baead68ab48fc4efd782b89946e0e0. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — no unsafe operations, secret leakage, or unauthorized access. <br>
- Correctness: Whether the final answer matches the reference diagnosis. <br>
- Discoverability: Whether the right skill was loaded and decoys avoided when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow (goal accuracy and behavior check, equally weighted). <br>
- Efficiency: Whether wasted tool calls and token usage were avoided (tool productivity and token efficiency, equally weighted). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Checks final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the user's goal was achieved. <br>
- `behavior_check`: Checks whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Checks tool-call productivity. <br>
- `token_efficiency`: Checks actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 98.1% (uplift unavailable) | 83.0% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 81.8% → 98.2% (+16.4 points) | 78.2% → 84.6% (+6.4 points) |
| Discoverability | 100.0% (uplift unavailable) | 95.0% (uplift unavailable) |
| Effectiveness | 70.4% → 94.4% (+24.0 points) | 65.1% → 79.5% (+14.4 points) |
| Efficiency | 97.7% (uplift unavailable) | 56.1% (uplift unavailable) |

## Skill Version(s): <br>
0b6e350b (source: git SHA, committed 2026-10-21) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
