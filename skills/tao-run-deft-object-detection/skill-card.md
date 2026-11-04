## Description: <br>
Run the full DEFT smart-data-augmentation loop for NVIDIA TAO Grounding DINO object detection: zero-shot baseline inference, KPI analysis, per-class gap analysis, SigLIP embedding of weak images, unique-neighbor mining against a source pool, ODVG dataset staging, and retraining — repeated for a fixed number of iterations. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who want to improve NVIDIA TAO Grounding DINO object-detection accuracy through automated smart data augmentation, including one-time source-pool preparation (Co-DETR pseudo-labeling, ODVG conversion, embedding), per-class gap analysis, unique-neighbor mining, and iterative retraining. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

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
**Output Type(s):** [Shell commands, Configuration instructions, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks; run artifacts (deft_state.json, loop_log.jsonl, DEFT_Loop_Report.md, checkpoints, KPI files)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), 1 attempt per task, each run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed. <br>
- Effectiveness: Whether the skill helped complete the task (50% goal accuracy + 50% behavior check). <br>
- Efficiency: Whether wasted tool calls and token usage were avoided (50% tool productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 94.2% — baseline ran, but no comparable score was available; uplift unavailable | 81.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 75.0% → 100.0% (+25.0 points) | 75.0% → 100.0% (+25.0 points) |
| Correctness | 10.0% → 100.0% (+90.0 points) | 20.0% → 75.0% (+55.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 71.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 5.1% → 87.6% (+82.5 points) | 21.4% → 67.0% (+45.6 points) |
| Efficiency | 83.2% — baseline ran, but no comparable score was available; uplift unavailable | 92.5% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
