## Description: <br>
Use when running the CDF AnomalyGen pipeline over a defect dataset — mask placement, fine-tuning, synthetic defect-image generation (SDG), evaluation, quality refinement, and pseudo-labeling — even when the user names only one stage. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building defect-detection systems use this skill to fine-tune the Cosmos base into an anomaly-generation model and produce synthetic defect images from clean images and defect masks, running any single stage or the full pipeline (mask placement, fine-tuning, generation, evaluation, refinement, and pseudo-labeling). <br>

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
- [Agentic mask generation reference](references/agentic-mask-generation.md) <br>
- [Fine-tuning reference](references/fine-tuning.md) <br>
- [Inputs reference](references/inputs.md) <br>
- [Mask-placement reference (AMP)](references/mask-placement.md) <br>
- [Quality-refinement reference](references/quality-refinement.md) <br>
- [Verification reference](references/verification.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks; pipeline artifacts on disk (masks, testcase.jsonl, checkpoints, generated images, KPI JSON, COCO pseudo-labels)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Each step is complete only once its verification gate confirms the output artifact on disk] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (5 positive), 1 attempt per task, each run in an isolated sandbox pod and compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? <br>
- Correctness: Is the answer correct? <br>
- Discoverability: Was the right skill loaded when needed? <br>
- Effectiveness: Did the skill help complete the task? <br>
- Efficiency: Did it avoid wasted tool calls and token usage? <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 71.4% — baseline ran, but no comparable score was available; uplift unavailable | 71.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 0.0% → 56.0% (+56.0 points) | 0.0% → 48.0% (+48.0 points) |
| Discoverability | 98.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 6.5% → 17.8% (+11.3 points) | 4.5% → 20.7% (+16.2 points) |
| Efficiency | 85.1% — baseline ran, but no comparable score was available; uplift unavailable | 95.7% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.2.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
