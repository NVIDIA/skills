## Description: <br>
Use this skill when answering a question about previously analyzed or freshly scoped VSS video, or when reading a stored VSS memory job or record by id, or whenever a question should be answered by running the `vss memory introspect` command; it routes through hot context, agent Markdown notes, `vss memory get` or `vss memory query`, `vss memory introspect`, or an exact-window `vss vlm run`, and is not for video retrieval or metadata-answerable questions. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and operators of a deployed NVIDIA Video Search and Summarization (VSS) blueprint use this skill to have an agent answer questions about analyzed or explicitly scoped video by routing through conversation context, agent Markdown memory, structured VSS memory, deployment-side introspection, or a single bounded VLM inspection via the `vss` CLI. <br>

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
- [AGENTS.md (vss CLI bootstrap, exit codes, and rules)](../../../AGENTS.md) <br>
- [BENCHMARK.md](BENCHMARK.md) <br>
- [VSS Documentation](https://docs.nvidia.com/vss/latest/index.html) <br>
- [NVIDIA AI Blueprint: Video Search and Summarization (GitHub)](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands] <br>
**Output Format:** [Natural-language answer grounded in `vss` CLI JSON results, with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Reports the `vss` CLI exit code alongside results and cites returned job/record handles; at most one VLM inspection per grounded scope.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
17 evaluation tasks (16 positive, 1 negative), one attempt per task, each run in its own isolated sandbox pod in a `k8s-sandbox` environment (evaluator version `1.6.9`, evaluated 2026-11-16, dataset digest `sha256:830efbe7c931304d5f442008bdbca6a2d7b5552ad59745dfd4190886fe681b13`); baseline is the same task attempted without the target skill. <br>

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
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.6% — baseline ran, but no comparable score was available; uplift unavailable | 83.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 97.1% → 100.0% (+2.9 points) | 91.2% → 100.0% (+8.8 points) |
| Correctness | 44.7% → 98.8% (+54.1 points) | 35.3% → 84.7% (+49.4 points) |
| Discoverability | 87.5% — baseline ran, but no comparable score was available; uplift unavailable | 81.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 41.7% → 74.0% (+32.3 points) | 43.1% → 58.3% (+15.2 points) |
| Efficiency | 97.8% — baseline ran, but no comparable score was available; uplift unavailable | 93.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 95.5% (-3.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.8% → 91.8% (-1.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Skill Version(s): <br>
3.3.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
