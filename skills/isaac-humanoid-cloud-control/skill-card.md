## Description: <br>
Run streamed Unitree G1 cloud-navigation demos in Isaac Sim with guarded gantry release. Use for demo, destination, Mission Control, or cleanup requests. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and robotics engineers use this skill to have an AI coding agent bring up the NVIDIA Isaac ROS release-4.6 Unitree G1 cloud-control stack (Mission Control, headless Isaac Sim with browser WebRTC streaming, and the G1 ROS launch), perform a guarded simulation-only gantry release, submit operator-approved navigation missions, and tear down demo-owned resources. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Other [Basic auth], API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [bootstrap.md](references/bootstrap.md) <br>
- [cloud-workspace.md](references/cloud-workspace.md) <br>
- [dependencies.md](references/dependencies.md) <br>
- [isaac-ros-setup.md](references/isaac-ros-setup.md) <br>
- [ordered-startup.md](references/ordered-startup.md) <br>
- [pinocchio-runtime.md](references/pinocchio-runtime.md) <br>
- [release-4.6.md](references/release-4.6.md) <br>
- [teardown.md](references/teardown.md) <br>
- [troubleshooting.md](references/troubleshooting.md) <br>
- [bring-up-cloud-stack/README.md](references/bring-up-cloud-stack/README.md) <br>
- [change-fleet-composition/README.md](references/change-fleet-composition/README.md) <br>
- [change-map/README.md](references/change-map/README.md) <br>
- [isaac-ros-activate/README.md](references/isaac-ros-activate/README.md) <br>
- [isaac-sim-installation/README.md](references/isaac-sim-installation/README.md) <br>
- [isaac-sim-remote/README.md](references/isaac-sim-remote/README.md) <br>
- [submit-and-monitor-mission/README.md](references/submit-and-monitor-mission/README.md) <br>
- [upstream-versions.lock.json](upstream-versions.lock.json) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, API Calls, Analysis] <br>
**Output Format:** [Markdown status report (State, Viewer, Mission, Cleanup) with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Mission submission and simulated gantry release require explicit operator approval; demo stack is kept running until an explicit teardown request.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
15 evaluation tasks (15 positive), 1 attempt per task, run in the `k8s-sandbox` environment with each task attempt in its own isolated sandbox pod. Each task was also attempted without the target skill as a baseline. Evaluator version `1.6.8`; evaluation date 2026-11-05. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored signals: `security` (100%). <br>
- Correctness: Is the answer correct? Scored signals: `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored signals: `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored signals: `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored signals: `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 84.2% — baseline ran, but no comparable score was available; uplift unavailable | 75.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 93.3% (-6.7 points) | 66.7% → 76.7% (+10.0 points) |
| Correctness | 29.3% → 88.0% (+58.7 points) | 22.7% → 81.3% (+58.6 points) |
| Discoverability | 98.3% — baseline ran, but no comparable score was available; uplift unavailable | 89.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 26.2% → 53.0% (+26.8 points) | 28.3% → 43.0% (+14.7 points) |
| Efficiency | 88.1% — baseline ran, but no comparable score was available; uplift unavailable | 85.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.2% → 91.5% (-5.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 91.2% → 87.4% (-3.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 64 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 15 task(s) |

## Skill Version(s): <br>
27033db (source: git SHA, committed 2026-11-05) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
