## Description: <br>
Detect and diagnose CUDA correctness bugs with NVIDIA Compute Sanitizer by routing GPU memory errors, shared-memory data races, uninitialized-memory reads, and synchronization/barrier bugs to the right sub-tool (memcheck, racecheck, initcheck, synccheck) and explaining how to build, run, and read the report. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers debugging CUDA GPU applications use this skill to detect and diagnose GPU-side correctness bugs, including illegal or out-of-bounds memory access, leaks, shared-memory races, uninitialized reads, and synchronization/barrier issues. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [memcheck.md](references/memcheck.md) <br>
- [racecheck.md](references/racecheck.md) <br>
- [initcheck.md](references/initcheck.md) <br>
- [synccheck.md](references/synccheck.md) <br>
- [changelog.md](references/changelog.md) <br>
- [Compute Sanitizer Release Notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html) <br>
- [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) <br>
- [NVIDIA Developer Tools Overview](https://developer.nvidia.com/tools-overview) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks; interprets Compute Sanitizer stdout/stderr reports with file:line localization] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
16 evaluation tasks (15 positive, 1 negative), 1 attempt per task, each run in an isolated k8s-sandbox pod and compared against a no-skill baseline. <br>

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
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill) | Codex (Baseline → Skill) |
|---|---:|---:|
| Overall | 96.9% (no comparable baseline) | 96.0% (no comparable baseline) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 96.3% → 98.8% (+2.5 points) | 96.3% → 100.0% (+3.7 points) |
| Discoverability | 93.0% (no comparable baseline) | 94.9% (no comparable baseline) |
| Effectiveness | 81.3% → 96.4% (+15.1 points) | 79.0% → 94.4% (+15.4 points) |
| Efficiency | 96.1% (no comparable baseline) | 90.8% (no comparable baseline) |

## Skill Version(s): <br>
489b2fb (source: git SHA, committed 2026-10-13) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
