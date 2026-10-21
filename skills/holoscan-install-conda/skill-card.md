## Description: <br>
Handle Holoscan SDK 5.x Conda installation requests: explain 5.0 package unavailability, check later releases, and select a supported installation method. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill when they ask to install Holoscan SDK 5.x with Conda, to learn that SDK 5.0 has no Conda package, check whether a later release documents Conda support, and choose a supported container, Debian, wheel, or source installation that preserves their requested SDK version and C++ or Python goal. <br>

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
- [Holoscan SDK Installation Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation) <br>
- [Holoscan SDK on GitHub](https://github.com/nvidia-holoscan/holoscan-sdk) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions] <br>
**Output Format:** [Markdown] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (5 positive, 1 negative), 1 attempt per task, each run in an isolated sandbox pod (`k8s-sandbox`), compared against a no-skill baseline. <br>

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
| Overall | 96.3% — no comparable baseline score; uplift unavailable | 94.7% — no comparable baseline score; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 83.3% (-16.7 points) |
| Correctness | 86.7% → 96.7% (+10.0 points) | 86.7% → 100.0% (+13.3 points) |
| Discoverability | 100.0% — no comparable baseline score; uplift unavailable | 95.0% — no comparable baseline score; uplift unavailable |
| Effectiveness | 87.1% → 85.8% (-1.3 points) | 74.6% → 97.1% (+22.5 points) |
| Efficiency | 99.0% — no comparable baseline score; uplift unavailable | 97.9% — no comparable baseline score; uplift unavailable |

Overall verdict: PASS (Tier 1 static validation passed with observations; Tier 2 semantic deduplication passed; Tier 3 live agent evaluation passed).

## Skill Version(s): <br>
2.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
