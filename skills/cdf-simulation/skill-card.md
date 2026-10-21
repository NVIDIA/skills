## Description: <br>
Component-owned guidance that turns one matched CDF Simulation request — submitted directly or through Airflow-triggered DIG Flow Catalog prompts — into one validated `cdf.CDFJob` build and its compiled JobOrder payload from a board USD, without executing a render itself. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers generating labeled synthetic PCB-AOI training data use this skill to translate a matched DIG Flow Catalog prompt or direct request into a validated, compiled CDF JobOrder payload for a caller-supplied board USD, which the caller's own executor then renders. <br>

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
- [class-filter-taxonomy.md](references/class-filter-taxonomy.md) <br>
- [fixed-defaults.md](references/fixed-defaults.md) <br>
- [payload-contract.md](references/payload-contract.md) <br>
- [reference-flows.md](references/reference-flows.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions] <br>
**Output Format:** [Python `cdf.CDFJob` build plus a compiled JobOrder payload (serialized with JobOrderCodec) and a resolved-inputs summary] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [One variant per payload, each marked as an independent launch; the skill compiles and returns the plan but never renders] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (8 positive), one attempt per task, each run in its own isolated k8s-sandbox pod against dataset snapshot `skill-evaluator-dataset-snapshot/1`. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the task, scored as the equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage, scored as 50% tool-call productivity and 50% token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 87.0% (uplift unavailable) | 76.7% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 12.5% → 92.5% (+80.0 points) | 0.0% → 55.0% (+55.0 points) |
| Discoverability | 95.0% (uplift unavailable) | 85.6% (uplift unavailable) |
| Effectiveness | 28.2% → 61.9% (+33.7 points) | 17.7% → 50.0% (+32.3 points) |
| Efficiency | 85.8% (uplift unavailable) | 93.1% (uplift unavailable) |

## Skill Version(s): <br>
1.0.0 (source: SKILL.md frontmatter metadata.version; repo pyproject.toml and README report package version 1.2.0) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
