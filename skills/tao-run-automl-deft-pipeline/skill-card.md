## Description: <br>
Run the canonical NVIDIA AOI three-phase training pipeline — Phase 1 AutoML baseline (HPO), Phase 2 DEFT loop (RCA → SDG → mining → plain-train retrain), Phase 3 AutoML refinement on the DEFT-augmented dataset. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run the full three-phase AOI training pipeline — AutoML hyperparameter optimization, DEFT data-improvement loop, and post-DEFT AutoML refinement — to fine-tune NVIDIA TAO ChangeNet models end-to-end. <br>

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
- [consolidated-preflight.md](references/consolidated-preflight.md) <br>
- [phase-handoffs.md](references/phase-handoffs.md) <br>
- [pitfalls-and-quality-checks.md](references/pitfalls-and-quality-checks.md) <br>
- [quick-start-example.md](references/quick-start-example.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 1 task (1 positive) from skill-evaluator-dataset-snapshot/1, with 3 attempts per task in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the final answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded and activated when needed. <br>
- Effectiveness: Checks whether the skill helped the agent complete the user's goal and follow the expected workflow. <br>
- Efficiency: Checks whether the skill avoided wasted tool calls and excessive token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies absence of unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Verifies tool-call productivity. <br>
- `token_efficiency`: Verifies actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 99.4% — baseline ran, but no comparable score was available; uplift unavailable | 79.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 6.7% → 100.0% (+93.3 points) | 20.0% → 100.0% (+80.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 0.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 16.7% → 100.0% (+83.3 points) | 48.3% → 100.0% (+51.7 points) |
| Efficiency | 97.2% — baseline ran, but no comparable score was available; uplift unavailable | 99.7% → 98.5% (-1.2 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
