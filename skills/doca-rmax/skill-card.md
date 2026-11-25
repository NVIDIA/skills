## Description: <br>
Guides agents through DOCA Rivermax (doca_rmax) receive streams for timing-precise media over IP (SMPTE ST 2110 video/audio, market data, scientific feeds) on a target BlueField DPU once the Rivermax libraries and license are installed, covering doca_rmax_in_stream sessions, the SDK + license precondition check, doca_rmax_get_*_supported capability queries, pairing with doca-eth queues and doca-flow steering, and DOCA_ERROR_* diagnosis from Rivermax calls, while routing queue programming, steering, PTP/clock sync, and Rivermax install/licensing to other skills. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
External developers and engineers building C/C++ (or FFI-bound) applications that call doca_rmax_* use this skill to configure, build, modify, run, test, and debug DOCA Rivermax receive streams for timing-precise media-over-IP workloads on a BlueField DPU with the Rivermax SDK and license already installed. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Other [separately obtained Rivermax software license file]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>
- [BENCHMARK.md](BENCHMARK.md) <br>
- [NVIDIA DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [DOCA Samples and Applications](https://github.com/NVIDIA-DOCA/doca-samples) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions, Code modification guidance] <br>
**Output Format:** [Markdown with inline bash and C code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), 1 attempt per task, each run in its own isolated sandbox pod (`k8s-sandbox` environment); evaluator version `1.6.10`, evaluation date 2026-11-23. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
**Overall verdict: PASS — Recommended for publication**

**Publication Recommendation:** Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 93.5% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 50.0% → 100.0% (+50.0 points) | 55.0% → 100.0% (+45.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 25.0% → 87.5% (+62.5 points) | 37.5% → 93.8% (+56.3 points) |
| Efficiency | 79.9% — baseline ran, but no comparable score was available; uplift unavailable | 69.6% — baseline ran, but no comparable score was available; uplift unavailable |

Baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points. A partial dimension was calculated from only the available configured signals.

### Comparison details

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.3% → 94.3% (-4.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.4% → 84.4% (±0.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 20 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Skill Version(s): <br>
444e267 (source: git SHA, committed 2026-11-23); targets DOCA 3.6 (source: frontmatter doca_version) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
