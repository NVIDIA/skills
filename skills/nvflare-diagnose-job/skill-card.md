## Description: <br>
Use when the user asks why a reported NVFLARE job failure signal occurred: the job failed, stalled, timed out, lost clients, ended with EXECUTION_EXCEPTION, or produced suspicious errors. Diagnose in simulation, POC, or production by collecting bounded evidence and mapping failure patterns to recovery actions. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers diagnosing NVFLARE federated learning job failures across simulation, POC, and production environments by collecting bounded evidence and mapping failure patterns to actionable recovery steps. <br>

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
- [Evidence Collection Guide](references/evidence-collection.md) <br>
- [Failure Pattern Catalog](references/failure-patterns.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Diagnosis report] <br>
**Output Format:** [Structured Markdown with runtime mode, matched failure pattern, recovery category, evidence summary, and next action] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 10 internal diagnosis tasks (10 positive) in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the diagnosis answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (goal completion and expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage, checking routing quality and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 68% → 82% (+15 points) |
| Security | Not available | 70% → 85% (+15 points) |
| Correctness | Not available | 94% → 96% (+2 points) |
| Discoverability | Not available | 41% → 69% (+28 points) |
| Effectiveness | Not available | 95% → 90% (-6 points) |
| Efficiency | Not available | 38% → 73% (+35 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
