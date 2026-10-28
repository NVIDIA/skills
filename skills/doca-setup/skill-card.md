## Description: <br>
Use this skill for the DOCA environment around a workload — verifying an install is healthy, preparing the build/runtime environment (pkg-config, headers, LD_LIBRARY_PATH, hugepages, devlink, representors), debugging environment-class failures, deciding container-vs-bare-metal deployment shape, and reaching a DOCA install from a host that does not have one yet via the NGC DOCA container Stage-1 fallback. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers working with NVIDIA DOCA on BlueField or host x86 use this skill to verify an install is healthy, prepare the build and runtime environment, diagnose environment-class setup failures, route deployment-shaped questions to the right container or bare-metal path, and reach a DOCA install from a host that does not yet have one. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [command-appendix.md](references/command-appendix.md) <br>
- [hardware-binding.md](references/hardware-binding.md) <br>
- [no-install.md](references/no-install.md) <br>
- [recognize.md](references/recognize.md) <br>
- [NVIDIA DOCA SDK documentation](https://docs.nvidia.com/doca/sdk/) <br>
- [NGC DOCA container tags](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/doca/containers/doca/tags) <br>
- [DOCA samples + applications](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [DOCA Platform Framework](https://github.com/NVIDIA/doca-platform) <br>
- [DOCA Developer Forum](https://forums.developer.nvidia.com/c/infrastructure/doca/370) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct against the reference answer? <br>
- Discoverability: Was the right skill loaded when needed, with decoys avoided? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 98.8% (uplift unavailable) | 91.4% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 65.0% → 100.0% (+35.0 points) | 60.0% → 95.0% (+35.0 points) |
| Discoverability | 100.0% (uplift unavailable) | 83.3% (uplift unavailable) |
| Effectiveness | 52.1% → 100.0% (+47.9 points) | 83.3% → 100.0% (+16.7 points) |
| Efficiency | 93.9% (uplift unavailable) | 78.7% (uplift unavailable) |

## Skill Version(s): <br>
c98e456 (source: git SHA, committed 2026-10-28); DOCA 3.5 (source: frontmatter doca_version) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
