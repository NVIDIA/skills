## Description: <br>
Choose a physics engine and integration for robot learning, manipulation, assembly, differentiable optimization, or other Physical AI simulation by comparing Newton, PhysX, MuJoCo/MJX, Bullet/PyBullet, VSim Robotics, Genesis, and Drake, evaluating Newton first for suitable new workflows, selecting ovnewton when supported shared-stage integration adds value, and then verifying setup. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building robot learning, manipulation, assembly, or differentiable-optimization simulations use this skill to choose a physics engine, solver, and integration path (such as direct Newton, Newton + ovnewton, or a framework-owned backend), resolve migration decisions, and verify Newton/ovnewton first-run setup. <br>

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
- [Newton physics engine (GitHub)](https://github.com/newton-physics/newton) <br>
- [Newton worlds concepts](https://newton-physics.github.io/newton/latest/concepts/worlds.html) <br>
- [Newton solvers and differentiation matrix](https://newton-physics.github.io/newton/latest/solvers/index.html) <br>
- [NVIDIA Isaac Lab](https://developer.nvidia.com/isaac/lab) <br>
- [Newton adds contact-rich manipulation and locomotion capabilities (NVIDIA blog)](https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline console code blocks; optional JSON metadata report from the environment probe script] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Recommendation states engine, solver, and integration separately, with closest alternative, capability conditions or unknowns, versions, checks actually run, and next setup action] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
18 evaluation tasks (16 positive, 2 negative), one attempt per task, each run in its own isolated sandbox pod (k8s-sandbox environment) with paired with-skill and without-skill baseline runs; SkillEvaluator version 1.5.6, evaluated 2026-10-14. Overall verdict: PASS — recommended for publication. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` (tool-call productivity) and 50% `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 95.6% — baseline ran, but no comparable score was available; uplift unavailable | 84.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 91.7% → 94.4% (+2.7 points) | 94.4% → 94.4% (±0.0 points) |
| Correctness | 90.0% → 100.0% (+10.0 points) | 97.8% → 100.0% (+2.2 points) |
| Discoverability | 98.8% — baseline ran, but no comparable score was available; uplift unavailable | 92.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 77.2% → 95.3% (+18.1 points) | 87.6% → 97.2% (+9.6 points) |
| Efficiency | 89.5% — baseline ran, but no comparable score was available; uplift unavailable | 36.6% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.2.0 (source: frontmatter, CHANGELOG) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
