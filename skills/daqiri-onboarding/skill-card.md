## Description: <br>
Guide DAQIRI setup and configuration generation or repair. Diagnose DAQIRI initialization failures, packet drops, RX-worker/application deficits, buffer errors and saved benchmark/counter evidence, including analysis-only requests with an existing runtime. Explain DAQIRI memory paths, select polling and RX reorder modes, and handle explicitly requested customer demos. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers working with NVIDIA DAQIRI use this skill to set up builds and runtimes, generate, edit or repair DAQIRI YAML configurations, run first benchmarks, choose polling and RX reorder modes, and diagnose initialization failures, packet drops, buffer errors and saved benchmark or counter evidence. <br>

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
- [Documentation and Helper Map](references/documentation-map.md) <br>
- [Configuration Generation and Repair](references/configuration.md) <br>
- [Evidence and Delivery Claims](references/evidence.md) <br>
- [First Benchmark](references/first-run.md) <br>
- [Setup](references/setup.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [Polling Modes](references/polling-modes.md) <br>
- [Reorder Selection](references/reorder-selection.md) <br>
- [ibverbs Initialization Limits](references/ibverbs-init-limits.md) <br>
- [DevX RX Gaps](references/ibverbs-rx-gaps.md) <br>
- [Optional Customer Demo](references/customer-demo.md) <br>
- [DAQIRI Documentation](https://nvidia.github.io/daqiri/) <br>
- [DAQIRI Getting Started](https://nvidia.github.io/daqiri/getting-started/) <br>
- [DAQIRI Benchmarking](https://nvidia.github.io/daqiri/benchmarks/) <br>
- [DAQIRI Concepts](https://nvidia.github.io/daqiri/concepts/) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions, Files, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks and YAML configuration files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Codex (model not recorded) <br>
- Claude (`claude-opus-5-5`) <br>



## Evaluation Tasks: <br>
14 evaluation tasks (14 positive), 3 attempts per task, run through an external BYOE (paired CLI) harness with read-only input fixtures; evaluator version `daqiri-paired-cli-byoe`, evaluation date 2026-11-25. Dataset digest and environment: not recorded (legacy or non-live result). Each task was also attempted without the skill as a baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Overall: Producer-reported overall score comparing the no-skill baseline with the skill-assisted run; its historical scoring formula is not recorded in this benchmark. <br>
- Security: Is it safe to use? Producer-reported dimension; its historical scoring formula is not recorded in this benchmark. Not available in this run. <br>
- Correctness: Is the answer correct? Producer-reported dimension; its historical scoring formula is not recorded in this benchmark. Not available in this run. <br>
- Discoverability: Was the right skill loaded when needed? Producer-reported dimension; its historical scoring formula is not recorded in this benchmark. Not available in this run. <br>
- Effectiveness: Did the skill help complete the task? Producer-reported dimension; its historical scoring formula is not recorded in this benchmark. Not available in this run. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Producer-reported dimension; its historical scoring formula is not recorded in this benchmark. Not available in this run. <br>



## Evaluation Results: <br>
**Reported evaluation verdict:** PASS (producer-reported; evaluation source: self-assessed) <br>
**CI Quality assessment:** SELF-ASSESSED <br>
**Publication status:** Publication eligible <br>
**Reported execution:** completed <br>

CI did not run Tier 3. Scores, comparisons, and evaluation verdicts below are producer-reported. Scoring definitions reference Skill Evaluator 1.6.10 and do not set the CI publication gate. <br>

Scoring definitions below are those of Skill Evaluator 1.6.10; older reports may use different formulas. Supplied scores are displayed unchanged. <br>

### Results at a Glance

| Measure | Codex (Baseline → Skill Uplift) | Claude (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Security | Not available | Not available |
| Correctness | Not available | Not available |
| Discoverability | Not available | Not available |
| Effectiveness | Not available | Not available |
| Efficiency | Not available | Not available |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 23 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Self-assessed evaluation | **SELF-ASSESSED** | Reported execution: completed; CI did not run Tier 3 |

Tier 3 validation was skipped and executed 0 checks. CI did not run Tier 3; evaluation source: self-assessed.

## Skill Version(s): <br>
e991d84 (source: git SHA, committed 2026-11-24) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
