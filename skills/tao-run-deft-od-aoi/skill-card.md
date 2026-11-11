## Description: <br>
Run a binary industrial-inspection DEFT loop with TAO RT-DETR: measure on frozen KPI/test roles, mine gap-similar real and clean images with SigLIP, accumulate admitted COCO data, retrain from one base checkpoint, and select by KPI AP50. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to iteratively improve a single-class (defect) AOI industrial-inspection detector with TAO RT-DETR, routing KPI false negatives and false positives to SigLIP-based real/clean image mining, optional AnomalyGenNext synthesis, cumulative COCO admission, and KPI-selected retraining; it is not intended for generic multiclass object detection. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [TAO Skill Bank Repository](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>
- [Inputs and packaged defaults](references/defaults.md) <br>
- [DEFT OD AOI data contract](references/data-contract.md) <br>
- [Normalized source handoff](references/source-manifest.md) <br>
- [Preflight and launch review](references/preflight.md) <br>
- [DEFT OD AOI pipeline](references/pipeline.md) <br>
- [Bundled scripts and delegated skills](references/scripts-and-agents.md) <br>
- [Gap matching and retrieval routing](references/gap-routing.md) <br>
- [Dual object-detection gap analysis](references/tao-analyze-gaps-od-map.md) <br>
- [RT-DETR training and selection policy](references/training-policy.md) <br>
- [AnomalyGenNext synthesis contract](references/anomalygen-pool.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks; nested YAML specs, JSON state/manifests, Parquet artifacts, and COCO JSON datasets] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Stage completion artifacts are hashed and recorded in deft_state.json and loop_log.jsonl] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 positive evaluation task (route-binary-deft-od-aoi), 1 attempt per task, each in an isolated k8s-sandbox pod; evaluator version 1.5.6, evaluated 2026-10-16. Overall verdict: PASS (Tier 1 passed with observations, Tier 2 passed, Tier 3 pass). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed. <br>
- Effectiveness: Whether the skill helped complete the task: equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Whether the run avoided wasted tool calls and token usage: 50% tool-call productivity and 50% token efficiency. <br>

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
| Overall | 92.9% — baseline ran, but no comparable score was available; uplift unavailable | 92.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 40.0% → 100.0% (+60.0 points) |
| Discoverability | 80.0% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 10.0% → 100.0% (+90.0 points) | 10.0% → 87.5% (+77.5 points) |
| Efficiency | 84.6% — baseline ran, but no comparable score was available; uplift unavailable | 83.0% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
