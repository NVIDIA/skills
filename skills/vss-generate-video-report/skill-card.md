## Description: <br>
Use this skill when producing a VSS analysis report — Mode A per-clip VLM, Mode B incident-range via video-analytics, Mode C SOP compliance via the SOP tools. Not for standalone video summarization, real-time alerts or ad-hoc Q&A. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and operators of an NVIDIA Video Search and Summarization (VSS) deployment use this skill to have an agent generate structured markdown reports: per-clip VLM video analysis reports, narrative incident-range reports from video analytics, and SOP compliance reports from the VA-MCP SOP tools. <br>

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
- [Video analysis report (Mode A)](references/report-types/video-analysis.md) <br>
- [Incident range report (Mode B)](references/report-types/incident-range.md) <br>
- [SOP compliance report (Mode C)](references/report-types/sop-compliance.md) <br>
- [Video Analysis Report template](references/report-templates/video-analysis-report.md) <br>
- [Incident Range Report template](references/report-templates/incident-range-report.md) <br>
- [SOP Compliance Report template](references/report-templates/sop-compliance-report.md) <br>
- [Default VLM prompt](references/default-vlm-prompt.md) <br>
- [VSS Documentation](https://docs.nvidia.com/vss/latest/index.html) <br>
- [VSS GitHub repository](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, API Calls, Shell commands] <br>
**Output Format:** [Markdown report (Video Analysis, Incident Range, or SOP Compliance template), produced via inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Reasoning (`<think>`) blocks are stripped from VLM output; browser-facing clip URLs are rewritten to the public VSS origin; empty-range results render as a single plain-text line.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
14 evaluation tasks (13 positive, 1 negative), 1 attempt per task, each run in its own isolated sandbox pod (`k8s-sandbox`), compared against a no-skill baseline (evaluator version `1.6.9`, evaluated 2026-11-16). <br>

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
| Overall | 83.3% — baseline ran, but no comparable score was available; uplift unavailable | 91.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 12.9% → 94.3% (+81.4 points) | 37.1% → 94.3% (+57.2 points) |
| Discoverability | 75.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 9.1% → 74.3% (+65.2 points) | 28.6% → 77.6% (+49.0 points) |
| Efficiency | 72.8% — baseline ran, but no comparable score was available; uplift unavailable | 95.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.6% → 94.1% (-4.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.2% → 91.1% (-1.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Skill Version(s): <br>
3.3.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
