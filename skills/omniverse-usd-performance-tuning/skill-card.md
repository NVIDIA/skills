## Description: <br>
Diagnose and optimize USD stages for slow loading, high memory, low FPS or validation; preserve asset boundaries and reuse shared definitions. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and technical artists working with OpenUSD and NVIDIA Omniverse use this skill to diagnose and optimize USD stages (for example, factory layouts) for faster loading, lower memory use or better rendering, while preserving required prim paths, appearance, placement, animation and interfaces and writing results to a separate output workspace. <br>

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
- [NVIDIA Factory-Level USD Structuring guide](https://docs.omniverse.nvidia.com/vfi/latest/guide/factory-level-structuring.html) <br>
- [NVIDIA CAD conversion guidance](https://docs.omniverse.nvidia.com/vfi/latest/guide/cad-conversion.html) <br>
- [Setup](references/setup.md) <br>
- [Structure and boundary guidance](references/structure.md) <br>
- [Asset authoring](references/asset-authoring.md) <br>
- [Operations](references/operations.md) <br>
- [Verification](references/verification.md) <br>
- [Library reuse](references/library-reuse.md) <br>
- [Standalone installation](references/install-standalone.md) <br>
- [Remote assets](references/remote-assets.md) <br>
- [Upstream USD Optimize guidance](references/upstreams/usd-optimize.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Files, Shell commands] <br>
**Output Format:** [Optimized USD assembly plus validated report JSON and generated Markdown and HTML reports, with a Markdown summary] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Writes only to a separate output workspace; source layers are not overwritten without explicit authorization. Live progress is exposed through a read-only browser dashboard.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
10 evaluation tasks (9 positive, 1 negative), one attempt per task, each run in an isolated sandbox pod; each task was also attempted without the skill as a baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal (100%). <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal (100%). <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` plus 50% `token_efficiency`. <br>

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
| Overall | 79.4% — baseline ran, but no comparable score was available; uplift unavailable | 78.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 90.0% (-10.0 points) | 90.0% → 100.0% (+10.0 points) |
| Correctness | 60.0% → 86.0% (+26.0 points) | 60.0% → 82.0% (+22.0 points) |
| Discoverability | 82.8% — baseline ran, but no comparable score was available; uplift unavailable | 76.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 36.6% → 60.4% (+23.8 points) | 36.6% → 57.2% (+20.6 points) |
| Efficiency | 77.6% — baseline ran, but no comparable score was available; uplift unavailable | 79.4% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.5.0 (source: frontmatter, CHANGELOG) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
