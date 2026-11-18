## Description: <br>
Connect the Newton physics engine to an Omniverse simulation or ovstage with ovnewton, covering importing a populated ovstage into Newton, stepping its physics, synchronizing stage edits, and publishing simulation state for Omniverse rendering. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers integrating the Newton physics engine with Omniverse use this skill to build Newton models from a populated ovstage, run an application-owned solver loop, consume sealed stage edits, and publish simulation state for rendering. <br>

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
- [ovnewton getting-started guide](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/docs/site/getting-started.md) <br>
- [ovnewton basic example](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/ovnewton/examples/example_ovnewton_basic.py) <br>
- [ovnewton robot example](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/ovnewton/examples/example_ovnewton_robot.py) <br>
- [ovnewton support guide](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/docs/site/support.md) <br>
- [ovnewton concepts guide](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/docs/site/concepts.md) <br>
- [ovnewton output-read guide](https://github.com/NVIDIA-Omniverse/ovnewton/blob/main/docs/site/output-reads.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (5 positive integration requests, 2 negative routing cases), one attempt per task, each run in an isolated sandbox pod and compared against a no-skill baseline. <br>

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
| Overall | 94.6% — baseline ran, but no comparable score was available; uplift unavailable | 88.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 78.6% → 92.9% (+14.3 points) | 85.7% → 85.7% (±0.0 points) |
| Correctness | 54.3% → 100.0% (+45.7 points) | 77.1% → 100.0% (+22.9 points) |
| Discoverability | 98.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 40.7% → 89.6% (+48.9 points) | 61.4% → 91.8% (+30.4 points) |
| Efficiency | 92.7% — baseline ran, but no comparable score was available; uplift unavailable | 71.0% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.2.0 (source: CHANGELOG, released 2026-10-08; pyproject.toml reports development version 0.3.0.dev0) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
