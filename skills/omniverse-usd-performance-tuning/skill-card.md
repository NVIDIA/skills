## Description: <br>
Diagnose and optimize USD stages for slow loading, high memory, low FPS or validation; preserve asset boundaries and reuse shared definitions. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and technical artists working with OpenUSD and NVIDIA Omniverse use this skill to diagnose and optimize USD stages (for example factory layouts) for loading time, memory use and rendering performance, running USD Optimize operations and validation while preserving asset boundaries, prim paths, appearance and downstream interfaces. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [OAuth Token, Service Account] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Factory-Level USD Structuring guide](https://docs.omniverse.nvidia.com/vfi/latest/guide/factory-level-structuring.html) <br>
- [CAD conversion guidance](https://docs.omniverse.nvidia.com/vfi/latest/guide/cad-conversion.html) <br>
- [Setup](references/setup.md) <br>
- [Standalone installation](references/install-standalone.md) <br>
- [Remote asset access](references/remote-assets.md) <br>
- [Structure and boundary guidance](references/structure.md) <br>
- [Asset authoring](references/asset-authoring.md) <br>
- [Library reuse](references/library-reuse.md) <br>
- [Operations](references/operations.md) <br>
- [Verification](references/verification.md) <br>
- [Upstream USD Optimize package guidance](references/upstreams/usd-optimize.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Files] <br>
**Output Format:** [Markdown with inline bash code blocks; optimized USD layers, JSON assessment/validation records, and Markdown/HTML optimization reports] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Source layers are not overwritten without explicit authorization; outputs are written to a separate output workspace with a read-only progress dashboard.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
10 evaluation tasks (9 positive, 1 negative), one attempt per task, each run in an isolated sandbox pod (`k8s-sandbox`), compared against a no-skill baseline. <br>

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
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 78.6% — baseline ran, but no comparable score was available; uplift unavailable | 76.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 90.0% → 90.0% (±0.0 points) |
| Correctness | 54.0% → 82.0% (+28.0 points) | 54.0% → 86.0% (+32.0 points) |
| Discoverability | 75.6% — baseline ran, but no comparable score was available; uplift unavailable | 76.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 33.9% → 53.4% (+19.5 points) | 40.8% → 56.6% (+15.8 points) |
| Efficiency | 82.1% — baseline ran, but no comparable score was available; uplift unavailable | 75.3% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.5.0 (source: frontmatter, CHANGELOG) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
