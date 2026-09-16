## Description: <br>
Run the full DEFT smart-data-augmentation loop for NVIDIA TAO Grounding DINO object detection: zero-shot baseline inference, KPI analysis, per-class gap analysis, SigLIP embedding of weak images, unique-neighbor mining against a source pool, ODVG dataset staging, and retraining — repeated for a fixed number of iterations. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who want to improve NVIDIA TAO Grounding DINO object-detection model accuracy through automated smart data augmentation, gap analysis, unique-neighbor mining, and iterative retraining. <br>

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
- [data-layout.md](references/data-layout.md) <br>
- [grounding-dino.md](references/grounding-dino.md) <br>
- [pipeline-and-state.md](references/pipeline-and-state.md) <br>
- [preflight.md](references/preflight.md) <br>
- [prep-source-pool.md](references/prep-source-pool.md) <br>
- [scripts-and-agents.md](references/scripts-and-agents.md) <br>
- [stage-mined-data.md](references/stage-mined-data.md) <br>
- [tao-analyze-detection-kpi.md](references/tao-analyze-detection-kpi.md) <br>
- [tao-analyze-gaps-od-map.md](references/tao-analyze-gaps-od-map.md) <br>
- [tao-generate-image-embeddings.md](references/tao-generate-image-embeddings.md) <br>
- [tao-mine-od-images.md](references/tao-mine-od-images.md) <br>
- [NVIDIA TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>
- [Agent Skills Open Standard](https://agentskills.io) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), 3 attempts per task, each in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.7% | 83.4% |
| Security | 83.3% → 100.0% (+16.7 pp) | 100.0% → 100.0% (±0.0 pp) |
| Correctness | 5.0% → 95.0% (+90.0 pp) | 24.0% → 100.0% (+76.0 pp) |
| Discoverability | 95.0% | 42.5% |
| Effectiveness | 10.4% → 86.0% (+75.6 pp) | 17.3% → 80.5% (+63.2 pp) |
| Efficiency | 82.6% | 94.2% |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
