## Description: <br>
Use this skill when first-time NeMo Relay users want to try Relay, choose the least-complex supported quick start, or verify initial value through the CLI, a maintained integration, or direct Python, Node.js, or Rust instrumentation before production setup. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers adopting NeMo Relay for the first time use this skill to select the simplest applicable quick-start path and verify observable Relay value before production deployment. <br>

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
- [CLI Try-Now Reference](references/cli-try-now.md) <br>
- [Built-In Integrations Try-Now Reference](references/built-in-integrations-try-now.md) <br>
- [Manual Language Try-Now Reference](references/manual-language-try-now.md) <br>
- [NeMo Relay Getting Started Quick Start](https://docs.nvidia.com/nemo/relay/dev/getting-started/quick-start) <br>
- [NeMo Relay Supported Integrations](https://docs.nvidia.com/nemo/relay/dev/supported-integrations/about) <br>
- [NeMo Relay Plugin Configuration](https://docs.nvidia.com/nemo/relay/dev/configure-plugins/about) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline bash and language-specific code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 15 tasks (14 positive, 1 negative) from the skill-evaluator dataset. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helps complete the user's goal and follows expected workflow behavior. <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage through good routing and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies absence of unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the expected skill was found and executed. <br>
- `skill_efficiency`: Verifies routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 49% → 88% (+39 points) | 51% → 83% (+32 points) |
| Security | 100% → 100% (±0 points) | 73% → 90% (+17 points) |
| Correctness | 23% → 92% (+69 points) | 68% → 89% (+21 points) |
| Discoverability | 51% → 95% (+44 points) | 44% → 90% (+45 points) |
| Effectiveness | 27% → 68% (+40 points) | 42% → 70% (+27 points) |
| Efficiency | 44% → 84% (+40 points) | 27% → 76% (+49 points) |

## Skill Version(s): <br>
7eb33b6 (source: git SHA, committed 2026-08-21) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
