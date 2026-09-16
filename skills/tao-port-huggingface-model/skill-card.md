## Description: <br>
Integrate a HuggingFace Computer Vision model into the NVIDIA TAO Toolkit ecosystem (tao-core config, tao-pytorch trainer, tao-deploy TensorRT pipeline). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to integrate HuggingFace Computer Vision models into the NVIDIA TAO Toolkit, covering model inspection, validation, training, ONNX export, TensorRT deployment, and container-based end-to-end testing. <br>

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
- [Phase 0 — Prerequisites](references/phase-0-prereqs.md) <br>
- [Phase 1 — Inspection](references/phase-1-inspection.md) <br>
- [Phase 2 — Codebase Exploration](references/phase-2-codebase.md) <br>
- [Phase 3 — Implementation](references/phase-3-implementation.md) <br>
- [Phase 4 — Deploy](references/phase-4-deploy.md) <br>
- [Phase 5 — Packaging](references/phase-5-packaging.md) <br>
- [Phase 6 — Container Tests](references/phase-6-container-tests.md) <br>
- [Phase 7 — Optimization](references/phase-7-optimization.md) <br>
- [HuggingFace Inspection Guide](references/hf-inspection.md) <br>
- [TAO Patterns](references/tao-patterns.md) <br>
- [Task Type Guide](references/task-type-guide.md) <br>
- [Cross-Cutting Concerns](references/cross-cutting.md) <br>
- [Workflow Consistency](references/workflow-consistency.md) <br>
- [NVIDIA TAO Skill Bank (GitHub)](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 1 task (1 positive) with 3 attempts per task in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Whether the user's goal was achieved and the expected workflow behavior was followed (equal-weight mean of goal completion and behavior check). <br>
- Efficiency: Tool-call productivity and token efficiency (equal-weight mean). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected and decoys were avoided. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 99.2% | 95.4% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 6.7% → 100.0% (+93.3 points) | 20.0% → 100.0% (+80.0 points) |
| Discoverability | 100.0% | 95.0% |
| Effectiveness | 16.7% → 100.0% (+83.3 points) | 38.3% → 83.3% (+45.0 points) |
| Efficiency | 95.8% | 98.7% |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
