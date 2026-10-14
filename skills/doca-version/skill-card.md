## Description: <br>
DOCA version handling: detect the installed release and validate the four-way match across pkg-config doca-common, applications/VERSION, doca_caps --version, and bfver/mlnx-release on BlueField; reason about NGC tags; look up whether a capability exists on this release; diagnose build-vs-runtime drift. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to detect installed DOCA versions, validate four-way version consistency across host and BlueField components, reason about NGC container tags, check capability availability per release, and diagnose build-vs-runtime drift. <br>

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
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>
- [DOCA Compatibility Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html) <br>
- [DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Measures final-answer accuracy against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? Measures whether the expected skill was selected and decoys were avoided. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Combines tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity; routing is scored under Discoverability. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.1% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 75.0% → 100.0% (+25.0 points) | 60.0% → 100.0% (+40.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 81.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 60.4% → 85.4% (+25.0 points) | 60.4% → 82.5% (+22.1 points) |
| Efficiency | 70.0% — baseline ran, but no comparable score was available; uplift unavailable | 94.4% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
7e143d8 (source: git SHA, committed 2026-10-12) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
