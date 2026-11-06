## Description: <br>
Use as the top-level router for Omniverse Realtime Viewer USD apps, XR delivery, SPG GPU processing, stylized rendering, and focused viewer reference documents. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to plan and build RTX-rendered OpenUSD viewer applications (browser-streamed, native desktop, XR, or cloud-deployed) by composing OVRTX rendering, OVStage scene state, and delivery/UI libraries, then adding capabilities such as picking, scene management, GPU processing, stylized rendering, and CAE/CFD visualization. <br>

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
- [references/routing.md](references/routing.md) <br>
- [references/usd-viewer-app/README.md](references/usd-viewer-app/README.md) <br>
- [references/dependencies/README.md](references/dependencies/README.md) <br>
- [references/conventions.md](references/conventions.md) <br>
- [references/ovstage-ovrtx-integration/README.md](references/ovstage-ovrtx-integration/README.md) <br>
- [references/viewer-ux-workflow/README.md](references/viewer-ux-workflow/README.md) <br>
- [references/validation.md](references/validation.md) <br>
- [references/troubleshooting/README.md](references/troubleshooting/README.md) <br>
- [references/spg-processing/README.md](references/spg-processing/README.md) <br>
- [references/stylized-visualization/README.md](references/stylized-visualization/README.md) <br>
- [references/xr-viewer-recipe/README.md](references/xr-viewer-recipe/README.md) <br>
- [references/cae-cfd-visualization/README.md](references/cae-cfd-visualization/README.md) <br>
- [references/cloud-deployment/README.md](references/cloud-deployment/README.md) <br>
- [NVIDIA-Omniverse/usd-brep](https://github.com/NVIDIA-Omniverse/usd-brep) <br>
- [NVIDIA-Omniverse/cae-openusd-plugins](https://github.com/NVIDIA-Omniverse/cae-openusd-plugins) <br>
- [NVIDIA/warp-simdata](https://github.com/NVIDIA/warp-simdata) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline code blocks (Python, TypeScript, C++, USDA, bash)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (7 positive, 1 negative), 1 attempt per task, each attempt run in its own isolated sandbox pod (`k8s-sandbox`); dataset digest `sha256:b6a6c8ae48e6631377089dd22912257405540a3a1fecd6db54bdcce1e550ee79`. Evaluator version `1.6.8`, evaluation date 2026-11-06. Baseline is the same task attempted without the target skill. <br>

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
> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.7% — baseline ran, but no comparable score was available; uplift unavailable | 88.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 93.8% (-6.2 points) |
| Correctness | 45.0% → 97.5% (+52.5 points) | 82.5% → 100.0% (+17.5 points) |
| Discoverability | 97.9% — baseline ran, but no comparable score was available; uplift unavailable | 89.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 38.5% → 81.3% (+42.8 points) | 69.8% → 85.9% (+16.1 points) |
| Efficiency | 86.8% — baseline ran, but no comparable score was available; uplift unavailable | 75.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 88.6% → 84.6% (-4.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.7% → 82.3% (-2.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 71 finding(s) |
| Tier 2 | Semantic deduplication | **INCOMPLETE** | Tier 2 scan incomplete: produced more than 1024 content chunks. |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Skill Version(s): <br>
0.3.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
