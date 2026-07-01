## Description: <br>
Build DeepStream GStreamer pipelines interactively by collecting pipeline requirements through an interactive questionnaire, then assembling the pipeline using a standalone BM25 retrieval backend with structural metadata boosting over 270+ verified pipelines. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to rapidly generate ready-to-run gst-launch-1.0 pipelines for NVIDIA DeepStream SDK video analytics workflows including object detection, tracking, and streaming on dGPU, Jetson, and SBSA platforms. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Assembly Rules](references/assembly-rules.md) <br>
- [Output Format](references/output-format.md) <br>
- [Requirement Extraction](references/requirement-extraction.md) <br>
- [Security and Limitations](references/security-and-limitations.md) <br>
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) <br>
- [DeepStream NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Tier 3 agent details not available in this report <br>



## Evaluation Tasks: <br>
Evaluated via NVSkills-Eval 3-Tier Evaluation with external profile. Tier 3 live agent evaluation not available in this report. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>



## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
