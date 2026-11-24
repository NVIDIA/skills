## Description: <br>
Run the disk-backed DEFT AOI improvement loop for NVIDIA Cosmos Reason 3 / Cosmos3 models, using Nano by default and Edge or Super when explicitly requested: evaluate the base model on Proxy and frozen Benchmark splits, mine real image pairs from Proxy gaps, generate AnomalyGen synthetic NG pairs, assemble a per-iteration Train JSON from both producers, train with Cosmos Framework LoRA SFT, and repeat through the selected platform's submit/status/logs/cancel contract, supporting bare OK/NG labels only. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers use this skill to iteratively improve a Cosmos Reason 3 (Cosmos3) model for bare OK/NG PCB automated optical inspection by running repeated evaluate, mine, synthesize, train, and re-evaluate cycles against a frozen Benchmark KPI gate on a user-selected compute platform. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [RCCA Report Template](references/RCCA_REPORT_TEMPLATE.md) <br>
- [Report Rendering](references/REPORT_RENDERING.md) <br>
- [Air-Gapped Cosmos3 DEFT AOI](references/air-gap.md) <br>
- [Bare OK/NG AOI Annotation](references/aoi-annotation.md) <br>
- [Cosmos Reason 3 Framework Action Contract](references/cosmos-reason.md) <br>
- [Cosmos3 DEFT AOI Data Layout](references/data-layout.md) <br>
- [Proxy RCCA and Benchmark KPI](references/gap-analysis.md) <br>
- [Frozen Benchmark Metric Contract](references/metric-contract.md) <br>
- [Network-Enabled Bootstrap](references/network-bootstrap.md) <br>
- [Cosmos3 DEFT Pipeline and State](references/pipeline-and-state.md) <br>
- [Cosmos3 DEFT AOI Pre-Flight](references/preflight.md) <br>
- [Cosmos3 DEFT Scripts and Agents](references/scripts-and-agents.md) <br>
- [Cosmos AnomalyGen Reference](references/tao-generate-anomalies.md) <br>
- [Cosmos3 AOI Real-Pair Mining](references/tao-mine-aoi-images.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks, TOML specs, JSON state/annotation files, and HTML/Markdown reports] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Run progress is persisted in a disk-backed deft_state.json state machine; GPU stages are submitted through the selected platform's submit/status/logs/cancel verbs after explicit user approval.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (3 positive), 1 attempt per task, each run in an isolated sandbox pod in the `k8s-sandbox` environment; each task was also attempted without the skill as a baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 88.7% — baseline ran, but no comparable score was available; uplift unavailable | 87.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 26.7% → 100.0% (+73.3 points) | 40.0% → 93.3% (+53.3 points) |
| Discoverability | 96.7% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 10.8% → 53.3% (+42.5 points) | 10.8% → 54.2% (+43.4 points) |
| Efficiency | 93.4% — baseline ran, but no comparable score was available; uplift unavailable | 95.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

### Comparison details

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.3% → 92.4% (-5.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.8% → 90.3% (+1.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 86 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 3 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

**Overall verdict:** PASS — Recommended for publication

**Publication Recommendation:** Recommended for publication based on the completed evaluation evidence in this report.

## Skill Version(s): <br>
0.2.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
