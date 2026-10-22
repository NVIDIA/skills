## Description: <br>
Locates authoritative NVIDIA DOCA information without access to the DOCA source tree — the docs.nvidia.com page for a library/service/tool, which DOCA libraries are installed and at what version, a sample on disk or GitHub, a path under /opt/mellanox/doca, or a renamed/404'd doc URL — and routes hands-on work to the owning skills. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers working with NVIDIA DOCA use this skill to find where the authoritative answer lives — public documentation entry points, the on-disk /opt/mellanox/doca install layout, installed library versions, and samples — and to recover renamed or 404'd documentation links, without browsing or hands-on execution. <br>

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
- [map.md — example question shapes, GitHub sources, install layout, version, URL audit, doc entry points](references/map.md) <br>
- [libraries.md — per-library public guides](references/libraries.md) <br>
- [services.md — in-bundle DOCA services](references/services.md) <br>
- [tools.md — in-bundle DOCA tools](references/tools.md) <br>
- [external-dpu.md — externally-productized DPU/service/firmware routing](references/external-dpu.md) <br>
- [external-fabric.md — fabric, orchestration, and switch product routing](references/external-fabric.md) <br>
- [NVIDIA DOCA SDK documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [DOCA samples and applications (GitHub)](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [DOCA Platform Framework (GitHub)](https://github.com/NVIDIA/doca-platform) <br>
- [DOCA Developer Forum](https://forums.developer.nvidia.com/c/infrastructure/doca/370) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (2 positive, 1 negative), each attempt run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct against the reference answer? <br>
- Discoverability: Was the right skill loaded when needed, and were decoys avoided? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected-workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Half tool-call productivity, half token efficiency. <br>

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
| Overall | 88.8% (uplift unavailable) | 84.6% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 46.7% → 86.7% (+40.0 points) | 53.3% → 86.7% (+33.4 points) |
| Discoverability | 100.0% (uplift unavailable) | 92.5% (uplift unavailable) |
| Effectiveness | 18.1% → 60.3% (+42.2 points) | 27.6% → 62.0% (+34.4 points) |
| Efficiency | 97.0% (uplift unavailable) | 81.9% (uplift unavailable) |

## Skill Version(s): <br>
3bc1c5d (source: git SHA, committed 2026-10-22) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
