## Description: <br>
Guides hands-on DOCA Management programming against BlueField / ConnectX devices — standing up management or representor contexts (doca_mgmt_dev_ctx / doca_mgmt_dev_rep_ctx), querying device caps, toggling congestion-control global status, modifying diagnostics data, setting ICM quotas, and issuing raw firmware commands via doca_mgmt_raw_cmd with the correct scope (CONFIGURATION / DEBUG_READ_ONLY / DEBUG_WRITE / DEBUG_WRITE_FULL), while routing mlxconfig, firmware reflash, streaming telemetry, doca_caps snapshots, and DOCA install to other skills. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers building fleet-management or orchestration tools use this skill to programmatically query and modify device-level state on BlueField DPUs and ConnectX NICs through the DOCA Management C API, including capability queries, feature-flag changes with rollback, ICM quotas, raw firmware commands, and debugging DOCA_ERROR_* results from doca_mgmt_* calls. <br>

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
- [NVIDIA DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [NVIDIA DOCA Samples and Applications](https://github.com/NVIDIA-DOCA/doca-samples) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash and C code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Does not author full DOCA Management application source; routes to shipped DOCA samples with minimum-diff modifications, and device-state changes follow the doca-hardware-safety policy.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), 1 attempt per task, each run in an isolated k8s-sandbox pod; evaluated with and without the skill (baseline). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? (security signal) <br>
- Correctness: Is the answer correct? (accuracy signal) <br>
- Discoverability: Was the right skill loaded when needed? (skill_execution signal) <br>
- Effectiveness: Did the skill help complete the task? (50% goal_accuracy + 50% behavior_check) <br>
- Efficiency: Did it avoid wasted tool calls and token usage? (50% skill_efficiency + 50% token_efficiency) <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.5% — baseline ran, but no comparable score was available; uplift unavailable | 83.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 87.5% (±0.0 points) |
| Correctness | 45.0% → 100.0% (+55.0 points) | 95.0% → 100.0% (+5.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 25.0% → 77.1% (+52.1 points) | 64.6% → 58.3% (-6.3 points) |
| Efficiency | 85.5% — baseline ran, but no comparable score was available; uplift unavailable | 74.2% — baseline ran, but no comparable score was available; uplift unavailable |

Overall verdict: PASS (evaluation date 2026-10-28, evaluator version 1.6.2).

## Skill Version(s): <br>
c98e456 (source: git SHA, committed 2026-10-28); targets DOCA 3.5 (source: frontmatter metadata) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
