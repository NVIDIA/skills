## Description: <br>
Use this skill when first-time NeMo Relay users want to try Relay, choose the least-complex supported quick start, or verify initial value through the CLI, a maintained integration, or direct Python, Node.js, or Rust instrumentation before production setup. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers onboarding to NeMo Relay who want to trial the framework, choose the quickest supported path to visible value, and verify initial instrumentation before committing to production setup. <br>

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
- [NeMo Relay CLI Overview](https://docs.nvidia.com/nemo/relay/dev/nemo-relay-cli/about) <br>
- [Supported Integrations](https://docs.nvidia.com/nemo/relay/dev/supported-integrations/about) <br>
- [Language Quick Starts](https://docs.nvidia.com/nemo/relay/dev/getting-started/quick-start) <br>
- [Plugin Configuration](https://docs.nvidia.com/nemo/relay/dev/configure-plugins/about) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Shell commands, Code] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
15 evaluation tasks (14 positive, 1 negative) in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the skill produces correct answers against reference answers. <br>
- Discoverability: Whether the right skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helps complete the user's goal and expected workflow. <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 48% → 86% (+38 points) | 51% → 79% (+29 points) |
| Security | 100% → 93% (-7 points) | 73% → 80% (+7 points) |
| Correctness | 25% → 95% (+69 points) | 63% → 88% (+25 points) |
| Discoverability | 49% → 95% (+46 points) | 48% → 89% (+41 points) |
| Effectiveness | 28% → 72% (+44 points) | 45% → 65% (+20 points) |
| Efficiency | 39% → 76% (+37 points) | 25% → 75% (+50 points) |

## Skill Version(s): <br>
db4ed2e (source: git SHA, committed 2026-08-12) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
