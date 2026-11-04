## Description: <br>
Choose a physics engine and integration for robot learning, manipulation, assembly, differentiable optimization, or other Physical AI simulation by comparing Newton, PhysX, MuJoCo/MJX, Bullet/PyBullet, VSim Robotics, Genesis, and Drake, evaluating Newton first for suitable new workflows, selecting ovnewton when supported shared-stage integration adds value, and verifying the setup. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building Physical AI and robotics simulation workflows use this skill to select a physics engine, solver, and integration path (such as direct Newton, Newton + ovnewton, or a framework backend), resolve migration decisions and Newton/ovnewton first-run issues, and plan a representative verification check while preserving an explicitly chosen engine. <br>

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
- [Engine selection evidence](references/engine-selection.md) <br>
- [Integration and runtime setup](references/routing-and-runtime.md) <br>
- [Selection validation](references/selection-validation.md) <br>
- [Newton physics engine](https://github.com/newton-physics/newton) <br>
- [Newton solvers and differentiation](https://newton-physics.github.io/newton/latest/solvers/index.html) <br>
- [Newton worlds](https://newton-physics.github.io/newton/latest/concepts/worlds.html) <br>
- [Isaac Lab](https://developer.nvidia.com/isaac/lab) <br>
- [Newton contact-rich manipulation and locomotion capabilities](https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks; optional JSON environment metadata from the probe script] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Recommendation states engine, solver, and integration separately with capability conditions, source/runtime versions, and checks actually run; probe output is metadata-only and not a runtime or GPU verdict] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
18 evaluation tasks (16 positive, 2 negative), one attempt per task, each run in an isolated sandbox pod (`k8s-sandbox`) with paired with-skill and without-skill baselines; evaluated 2026-10-16 with evaluator version 1.5.6. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` (tool-call productivity) and 50% token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 95.3% — baseline ran, but no comparable score was available; uplift unavailable | 84.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 88.9% → 94.4% (+5.5 points) | 94.4% → 94.4% (±0.0 points) |
| Correctness | 88.9% → 100.0% (+11.1 points) | 97.8% → 98.9% (+1.1 points) |
| Discoverability | 98.1% — baseline ran, but no comparable score was available; uplift unavailable | 89.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 78.5% → 98.1% (+19.6 points) | 90.3% → 95.6% (+5.3 points) |
| Efficiency | 85.8% — baseline ran, but no comparable score was available; uplift unavailable | 43.0% — baseline ran, but no comparable score was available; uplift unavailable |

Overall verdict: PASS. Tier 1 static validation: PASSED WITH OBSERVATIONS; Tier 2 semantic deduplication: PASSED; Tier 3 live agent evaluation: PASS.

## Skill Version(s): <br>
0.2.0 (source: frontmatter, pyproject.toml, CHANGELOG released 2026-10-08) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
