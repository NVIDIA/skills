## Description: <br>
Use this skill when migrating applications, examples, integrations, documentation, manifests, or repository code from NeMo Flow to NeMo Relay across Python, Rust, Node.js, Go, C FFI, CLI, configuration, and observability surfaces. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers migrating existing NeMo Flow applications, libraries, or documentation to NeMo Relay across multiple language surfaces. <br>

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
- [NeMo Relay GitHub Repository](https://github.com/NVIDIA/NeMo-Relay/) <br>
- [Apache 2.0 License](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Code, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 5 tasks (4 positive, 1 negative) in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage. <br>

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
| Overall | 55% → 81% (+26 points) | 46% → 73% (+28 points) |
| Security | 100% → 100% (±0 points) | 80% → 100% (+20 points) |
| Correctness | 44% → 92% (+48 points) | 44% → 60% (+16 points) |
| Discoverability | 32% → 91% (+59 points) | 32% → 85% (+52 points) |
| Effectiveness | 51% → 28% (-22 points) | 33% → 29% (-4 points) |
| Efficiency | 46% → 92% (+46 points) | 39% → 93% (+54 points) |

## Skill Version(s): <br>
db4ed2e (source: git SHA, committed 2026-08-12) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
