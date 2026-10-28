## Description: <br>
Locates authoritative NVIDIA DOCA information without the source tree — the docs.nvidia.com page for a library/service/tool, which DOCA libraries are installed and at what version, a sample on disk or GitHub, a path under /opt/mellanox/doca, or a renamed/404'd doc URL — and routes hands-on programming, environment preparation, API tutorials, and hardware/firmware mutation to the owning skills. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers working with NVIDIA DOCA use this skill to locate authoritative information — the correct docs.nvidia.com page, the installed /opt/mellanox/doca layout and library versions, on-disk or GitHub samples, and recovery for renamed or 404'd doc URLs — and to route hands-on tasks to the owning workflow skills. <br>

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
- [map.md — example question shapes, GitHub sources, on-disk layout, version, URL audit, doc entry points](references/map.md) <br>
- [libraries.md — per-library public guides](references/libraries.md) <br>
- [services.md — in-bundle DOCA services](references/services.md) <br>
- [tools.md — in-bundle DOCA tools](references/tools.md) <br>
- [external-dpu.md — externally-productized DPU/service/firmware routing](references/external-dpu.md) <br>
- [external-fabric.md — fabric, orchestration, and firmware product routing](references/external-fabric.md) <br>
- [DOCA SDK documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [DOCA samples and applications (GitHub)](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [DOCA Platform Framework (GitHub)](https://github.com/NVIDIA/doca-platform) <br>
- [DOCA Developer Forum](https://forums.developer.nvidia.com/c/infrastructure/doca/370) <br>


## Skill Output: <br>
**Output Type(s):** [Documentation references, Shell commands] <br>
**Output Format:** [Markdown with inline URLs and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (2 positive, 1 negative), 1 attempt per task, each run in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Checks final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? Checks skill selection, decoy avoidance, and workflow execution. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 90.1% — baseline ran, but no comparable score was available; uplift unavailable | 83.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 40.0% → 86.7% (+46.7 points) | 86.7% → 86.7% (±0.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 27.8% → 66.7% (+38.9 points) | 50.0% → 66.7% (+16.7 points) |
| Efficiency | 97.4% — baseline ran, but no comparable score was available; uplift unavailable | 70.0% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
259da15 (source: git SHA, committed 2026-10-28); targets DOCA 3.5 (source: frontmatter metadata.doca_version) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
