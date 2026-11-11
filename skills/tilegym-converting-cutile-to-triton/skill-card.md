## Description: <br>
Converts cuTile GPU kernels (@ct.kernel) to Triton (@triton.jit), covering standard in-repo conversion, debugging (cudaErrorIllegalAddress, shape mismatch, numerical mismatch), mapping cuTile idioms (ct.load/ct.store, ct.Constant, ct.launch) to Triton equivalents, and dual-kernel layout flags with autotune grids via META. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers working on GPU kernels use this skill to port, translate, and optimize cuTile kernels in the TileGym repository as Triton kernels, and to debug existing Triton translations through a phase-gated analyze, convert, validate, test, and benchmark workflow. <br>

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
- [cuTile → Triton API Mapping](references/api-mapping.md) <br>
- [Triton Runtime Error Debugging (cuTile → Triton)](references/debugging.md) <br>
- [Gotchas — Most Common cuTile → Triton Translation Errors](references/gotchas.md) <br>
- [Testing & Validation (cuTile → Triton)](references/harness-integration.md) <br>
- [Optimization strategy (cuTile → Triton)](references/optimization-strategy.md) <br>
- [Triton Optimization Reference (cuTile → Triton)](references/optimizing-reference.md) <br>
- [Performance Gotchas — 10-50× Regression Risk](references/performance-gotchas.md) <br>
- [cuTile to Triton Conversion Workflow](translations/workflow.md) <br>
- [File Structure & Registration (cuTile → Triton)](translations/file-structure.md) <br>
- [Advanced Triton patterns (cuTile → Triton)](translations/advanced-patterns.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, Analysis] <br>
**Output Format:** [Python source files (Triton kernels and host wrappers) plus Markdown with inline bash code blocks and performance comparison tables] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (1 positive, 4 negative), 1 attempt per task, each run in an isolated sandbox pod; evaluated through a three-tier evaluation (static validation, semantic deduplication, live agent evaluation). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of `skill_efficiency` and `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 99.8% — baseline ran, but no comparable score was available; uplift unavailable | 98.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 80.0% → 100.0% (+20.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 82.5% → 100.0% (+17.5 points) | 90.0% → 99.0% (+9.0 points) |
| Efficiency | 99.2% — baseline ran, but no comparable score was available; uplift unavailable | 97.6% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.0.0 (source: frontmatter metadata.version) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
