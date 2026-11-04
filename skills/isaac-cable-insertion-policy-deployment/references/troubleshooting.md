# Troubleshooting

Preserve the failed session. Read the command's named log and correct a concrete
cause before starting a new session. Do not repeatedly run GPU jobs with the same
unresolved error. The driver stops its own processes and both containers on failure.

| Symptom | Response |
| --- | --- |
| Simulator image missing or wrong build | Pull the documented Isaac Sim 6.0.0 image or select a compatible prepared image with `--sim-image`. Inspect `sim-image.json` and `sim-version.txt`; do not substitute another release. |
| Missing ffmpeg or libx264 | Use `--install-sim-runtime` with a new invocation, or prepare the simulator image with `/usr/bin/ffmpeg`. The encoder check runs in that container. |
| Compatibility result absent or FAILED | Inspect `compatibility.log`, GPU driver and Docker GPU access. Exit code alone does not establish compatibility. |
| Simulator executable missing or not executable | The image is incomplete or modified. Select the official image and rerun in a new invocation; inspect `sim-preflight.log`. |
| Missing saved YAML, incompatible task or recurrent actor | Stop and request the originating parameters or supported checkpoint. Do not infer them from a different run. |
| Missing LEAPP/RSL-RL or export error | Inspect `prepare.log`/`export.log`; use the locked export-group installation and GPU-provider checks in [setup](setup.md). Do not patch source or load a different checkpoint without recording the change. |
| Reset handler reports `slice has no len` | The bundled export wrapper normalizes reset selectors to int32 tensor indices only for reset events while the upstream exporter runs, then restores the method. This changes neither policy semantics nor repository files. |
| Reset handler references `_get_delta_dof_pos` | The wrapper supplies a temporary alias to IsaacLab 3.0's renamed `get_delta_dof_pos`, then removes it after export. |
| Missing CUPTI or cuBLAS/cuFFT | Install binary `cuda-cupti-13-2` in the owned container and expose the existing NVIDIA cu13 wheel library directory. These fixes are included in the driver. |
| APT package/image unavailable | Check NVIDIA repository access and version; report it. Do not substitute a source checkout silently. |
| Policy response timeout or future timestamp | Inspect `policy-sync.jsonl`, simulator/policy logs and the dedicated ROS domain. The scene holds physics at 30 Hz boundaries; an unavailable or mismatched response invalidates the trial. Do not disable synchronization or add a fixed sleep to obtain success. |
| No ROS observations/targets | Check domain, Fast DDS UDPv4, scene playback, enable gate and Triton readiness. |
| Scene script layout changed | Stop on an unrecognized Jacobian/controller patch point. Inspect the new published scene before adapting it. |
| USD or material parity fails | Resolve the exact saved training URL and compare the report. Do not fall back to visually similar assets. |
| Empty RGB/video | Renderer warmup must complete before creating the grasp. The recorder requires a real 1280×720 uint8 RGB(A) frame before proceeding. |
| Prepared grasp fails verification | Inspect position/orientation error, clearance and gripper closure. Resolve setup ordering; do not count this as a policy trial or relax contact physics. |
| GPU OOM | Close only owned stale processes, check GPU use, and retry one headless environment. Do not change training geometry or contact settings. |
| Valid attempt fails insertion | Report measured failure and retain video. Follow the [trial and retry options](setup.md#trial-and-retry-options) for bounded unchanged repeats; controller/policy tuning is a separate task. |

Never describe a completed script or a playable video alone as insertion success.
