## Description: <br>
Integrates a HuggingFace Computer Vision model into the NVIDIA TAO Toolkit ecosystem (tao-core config, tao-pytorch trainer, tao-deploy TensorRT pipeline) through a 7-phase loop covering prerequisites, HuggingFace inspection and validation, codebase exploration, native trainer implementation, ONNX export and TensorRT deploy integration, packaging and L0 testing, container-based end-to-end validation, and conditional accuracy/latency tuning for classification, detection, segmentation, zero-shot detection, and depth estimation models. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to port a HuggingFace Computer Vision model (by model ID or URL) into TAO Toolkit, producing a tao-core config schema, a tao-pytorch trainer with evaluation and inference, ONNX export, a tao-deploy TensorRT engine pipeline, packaging, and L0 tests validated end to end inside local TAO Toolkit containers. <br>

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
- [phase-0-prereqs.md](references/phase-0-prereqs.md) <br>
- [phase-1-inspection.md](references/phase-1-inspection.md) <br>
- [hf-inspection.md](references/hf-inspection.md) <br>
- [phase-2-codebase.md](references/phase-2-codebase.md) <br>
- [task-type-guide.md](references/task-type-guide.md) <br>
- [phase-3-implementation.md](references/phase-3-implementation.md) <br>
- [tao-patterns.md](references/tao-patterns.md) <br>
- [repo-structure.md](references/repo-structure.md) <br>
- [phase-4-deploy.md](references/phase-4-deploy.md) <br>
- [phase-5-packaging.md](references/phase-5-packaging.md) <br>
- [phase-6-container-tests.md](references/phase-6-container-tests.md) <br>
- [docker-patterns.md](references/docker-patterns.md) <br>
- [phase-7-optimization.md](references/phase-7-optimization.md) <br>
- [workflow-consistency.md](references/workflow-consistency.md) <br>
- [cross-cutting.md](references/cross-cutting.md) <br>
- [TAO Skill Bank (GitHub)](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks, plus Python source, YAML specs, and local Docker invocations] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [All work stays local; no pushes to remotes or image publishing] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive), 1 attempt per task, run in isolated sandbox pods in the `k8s-sandbox` environment (evaluator version `1.6.10`, evaluated 2026-11-20); each task was also attempted without the skill as a baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` (50%) and `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? `skill_efficiency` (50%) plus `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 99.5% — baseline ran, but no comparable score was available; uplift unavailable | 61.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 20.0% → 100.0% (+80.0 points) | 20.0% → 80.0% (+60.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 0.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 33.3% → 100.0% (+66.7 points) | 33.3% → 33.3% (±0.0 points) |
| Efficiency | 97.5% — baseline ran, but no comparable score was available; uplift unavailable | 93.8% → 93.6% (-0.2 points) |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 99.1% → 94.9% (-4.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 30 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 25 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 1 task(s) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
