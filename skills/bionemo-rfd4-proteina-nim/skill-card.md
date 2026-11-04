## Description: <br>
Run RFD4-Proteina NIM for joint protein structure and sequence design, conditioned generation, and reward-guided search using the NVIDIA hosted API or a self-hosted NIM. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and scientists use this skill to design proteins with the RFD4-Proteina NIM — conditioned protein and small-molecule binders, motif and enzyme scaffolding, antibody/CDR design, partial diffusion, inverse folding, and reward-guided beam search — running against the NVIDIA hosted API or a self-hosted NIM and inspecting the returned structure and sequence candidates. <br>

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
- [deployment.md](references/deployment.md) <br>
- [examples.md](references/examples.md) <br>
- [parameters.md](references/parameters.md) <br>
- [request-api.md](references/request-api.md) <br>
- [science.md](references/science.md) <br>
- [sources.md](references/sources.md) <br>
- [validation.md](references/validation.md) <br>
- [RFD4-Proteina NIM product page](https://build.nvidia.com/ipd/rfd4-proteina) <br>
- [RFD4-Proteina NGC container listing](https://catalog.ngc.nvidia.com/orgs/nim/teams/ipd/containers/rfd4-proteina) <br>


## Skill Output: <br>
**Output Type(s):** [Files, API Calls, Analysis] <br>
**Output Format:** [JSON, mmCIF (.cif), and FASTA run artifacts with a summary report] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Per-design metrics, origins, diagnostics, and model/runtime/schedule provenance saved to a chosen run directory; generated structures are unvalidated design candidates] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (5 positive) spanning the condition-spec, hosted-binder, local-public-deployment, offline-search, and oracle-and-pending cases; one attempt per task, each run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct against the reference answer? <br>
- Discoverability: Was the right skill loaded, decoys avoided, and the workflow executed when needed? <br>
- Effectiveness: Did the skill help complete the task, measured as the equal-weight mean of goal completion and expected workflow adherence? <br>
- Efficiency: Did it avoid wasted tool calls and token usage, measured as equal-weight tool-call productivity and token efficiency? <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 94.3% (uplift unavailable) | 82.6% (uplift unavailable) |
| Security | 90.0% → 100.0% (+10.0 points) | 70.0% → 80.0% (+10.0 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 76.0% → 88.0% (+12.0 points) |
| Discoverability | 98.6% (uplift unavailable) | 90.0% (uplift unavailable) |
| Effectiveness | 43.5% → 84.0% (+40.5 points) | 44.5% → 76.5% (+32.0 points) |
| Efficiency | 88.7% (uplift unavailable) | 78.6% (uplift unavailable) |

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
