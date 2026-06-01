## Description: <br>
Use to run AutoMagicCalib on local MP4s, RTSP, or the bundled sample dataset, and to deploy vss-auto-calibration when needed. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run automated camera calibration (AutoMagicCalib) on video inputs from local files, RTSP streams, or bundled sample datasets, and to deploy the AMC microservice when needed. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NVIDIA VSS Documentation](https://docs.nvidia.com/vss/latest/index.html) <br>
- [Video Search and Summarization GitHub](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) <br>
- [Deploy Auto-Calibration Service](references/deploy-auto-calibration-service.md) <br>
- [Videos Input Mode](references/videos.md) <br>
- [RTSP Input Mode](references/rtsp.md) <br>
- [Sample Dataset](references/sample-dataset.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, API Calls, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks and JSON API payloads] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>



## Skill Version(s): <br>
3.2.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
