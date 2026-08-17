## Description: <br>
Use when measuring Jetson Video Codec SDK or PyNvVideoCodec encode/decode throughput, comparing presets or surfaces, testing codec-worker capacity with authenticated samples and user media, or producing a documented clock-scaled or clock-and-resolution-scaled planning estimate when representative content is unavailable. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers measuring codec-stage throughput on NVIDIA Jetson devices for video pipeline planning, capacity testing, and performance comparison across encode/decode presets and surfaces. <br>

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
- [Benchmark workflow](references/benchmark-workflow.md) <br>
- [Benchmark output contract](references/benchmark-output-contract.md) <br>
- [Documented performance estimates](references/documented-performance-estimates.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, JSON] <br>
**Output Format:** [Structured JSON results with Markdown presentation] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Results include per-repetition FPS, MP/s, mean/min/max statistics, and provenance metadata] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive) run in isolated sandbox pods with dataset digest sha256:e2d80ff3bff2819a83593a6167c1fd9119caa0d8febe727d4a947e1b84baec88. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Validates final-answer correctness against reference answers. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (goal completion and expected workflow adherence). <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use without wasted skill or tool usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 47% → 97% (+50 points) | 49% → 92% (+43 points) |
| Security | 100% → 100% (±0 points) | 86% → 100% (+14 points) |
| Correctness | 51% → 100% (+49 points) | 69% → 83% (+14 points) |
| Discoverability | 28% → 100% (+72 points) | 48% → 94% (+46 points) |
| Effectiveness | 36% → 84% (+48 points) | 32% → 90% (+58 points) |
| Efficiency | 19% → 100% (+81 points) | 11% → 92% (+82 points) |

## Skill Version(s): <br>
e61c045 (source: git SHA, committed 2026-08-10) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
