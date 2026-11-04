## Description: <br>
Agent runbook for evaluating an existing PDB directory through the separately installed first-party Proteina-Complexa CLI to refold or score designs, compute interface pAE, i_pLDDT, scRMSD, motif RMSD, or designability, run complexa analysis, or assess third-party BindCraft, AlphaProteo, or RFdiffusion outputs. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
## Use Case: <br>
Developers and computational biologists use this skill to score a directory of pre-existing PDB files against the same metrics Proteina-Complexa uses internally — selecting the correct evaluate/analyze configuration, running a GPU refold (AF2, RF3, ESMFold, or Boltz2), computing interface and monomer metrics, applying result-type thresholds, and summarizing per-design success and diversity results with a replayable run manifest. <br>

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
- [Evaluate Config Matrix](references/eval_configs.md) <br>
- [Evaluation Metrics Guide](references/EVALUATION_METRICS.md) <br>
- [Pipeline Configuration Guide](references/CONFIGURATION_GUIDE.md) <br>
- [Inference and Search Guide](references/INFERENCE.md) <br>
- [Search Metadata Tags](references/SEARCH_METADATA.md) <br>
- [Sweep System](references/SWEEP.md) <br>
- [Hardware Reference](references/hardware.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks, plus a JSON run manifest and result CSV summaries] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Writes a replayable JSON run manifest and summarizes per-PDB metrics, pass-rate, and diversity CSVs produced by the CLI run] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each run in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is the skill safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the final answer correct against the reference answer? <br>
- Discoverability: Was the expected skill selected and activated when needed while decoys were avoided? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did the run avoid wasted tool calls and token usage? <br>

Underlying evaluation signals used in this run: <br>
- `security`: Flags unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 87.7% — baseline ran, but no comparable score was available; uplift unavailable | 82.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 75.0% → 100.0% (+25.0 points) |
| Correctness | 25.0% → 100.0% (+75.0 points) | 40.0% → 100.0% (+60.0 points) |
| Discoverability | 90.0% — baseline ran, but no comparable score was available; uplift unavailable | 88.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 27.1% → 61.3% (+34.2 points) | 36.3% → 46.3% (+10.0 points) |
| Efficiency | 87.0% — baseline ran, but no comparable score was available; uplift unavailable | 77.5% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.1.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
