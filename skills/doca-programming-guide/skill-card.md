## Description: <br>
Guides developers through library-agnostic DOCA programming tasks including writing a first app from a shipped sample, the canonical pkg-config + meson build pattern, the Core object lifecycle, DOCA_ERROR decoding, and FFI/bindings for non-C consumers. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers building applications that consume DOCA libraries use this skill to follow library-agnostic programming patterns — from deriving a first app from shipped samples, through build configuration and the Core lifecycle, to error decoding and program-class debugging. <br>

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
- [DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/) <br>
- [DOCA Samples (GitHub)](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [DOCA Platform Framework (GitHub)](https://github.com/NVIDIA/doca-platform) <br>
- [NGC Catalog](https://catalog.ngc.nvidia.com/) <br>
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 tasks (3 positive, 1 negative) in isolated sandbox pods; dataset digest sha256:2e27fd34. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and the expected workflow behavior was followed (goal_accuracy 50% + behavior_check 50%). <br>
- Efficiency: Checks tool-call productivity and token efficiency (skill_efficiency 50% + token_efficiency 50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies whether the expected skill was selected and decoys were avoided. <br>
- `accuracy`: Evaluates final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Assesses whether the user's goal was achieved. <br>
- `behavior_check`: Assesses whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Measures tool-call productivity. <br>
- `token_efficiency`: Measures token usage efficiency. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill) | Codex (Baseline → Skill) |
|---|---:|---:|
| Overall | 90.6% | 93.5% |
| Security | 100.0% → 100.0% (±0.0) | 100.0% → 100.0% (±0.0) |
| Correctness | 60.0% → 100.0% (+40.0) | 80.0% → 95.0% (+15.0) |
| Discoverability | 100.0% | 93.3% |
| Effectiveness | 41.7% → 84.2% (+42.5) | 59.6% → 87.5% (+27.9) |
| Efficiency | 68.9% | 91.7% |

## Skill Version(s): <br>
6e4ede9 (source: git SHA, committed 2026-10-05) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
