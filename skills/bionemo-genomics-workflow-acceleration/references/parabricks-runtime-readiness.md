# Parabricks runtime readiness (workflow acceleration)

Run a lightweight runtime check **early** (SKILL.md §2 — Runtime readiness) before
promising accelerated runs on the user's current machine. **Inspection and in-place
wiring** can continue when Parabricks is not runnable locally; **execution** and
toggle-off vs toggle-on comparison need a GPU environment.

## Quick check

Run only what the user's environment allows (avoid long downloads without consent).

| Check | Command / action | Pass signal |
|-------|------------------|-------------|
| NVIDIA driver + GPU visible | `nvidia-smi` | Lists GPU(s), driver version |
| Parabricks CLI | `pbrun --version` or `pbrun version` | Version string, no "command not found" |
| Container (optional) | `docker run --rm --gpus all <parabricks-image> pbrun --version` | Version inside container; pin image tag in notes |

Record in the report: `Runtime: local ready | local not ready | unknown (not checked)`.

**Not runnable here** if: no GPU in `nvidia-smi`, no `pbrun`, no working Parabricks container, or user states laptop/CI without GPU.

## If not runnable — ask the user

Do not skip inspection; separate **design** from **where to run**.

Ask whether the user has GPU resources elsewhere: shared HPC, cloud (AWS/GCP/Azure/OCI), both, or not yet. Record where GPU runs will happen when known.

## Detailed runtime assessment

For full GPU/driver/Docker/container/storage readiness, installation guidance, and
per-tool commands, use the **`parabricks` skill**:

- [skills/parabricks/references/runtime-environment.md](../../parabricks/references/runtime-environment.md)
- Diagnostic script: `bash skills/parabricks/scripts/check_parabricks_runtime.sh`

## Document in deliverables

In **ACCELERATION.md**:

```markdown
## Runtime target
- Local Parabricks ready: yes/no
- Intended execution: HPC | AWS | GCP | Azure | other | TBD
- Notes: partition, region, image tag, executor profile name
```

## Agent behavior

| Situation | Behavior |
|-----------|----------|
| Local ready | Proceed with implement + compare on local/same host |
| Local not ready, user has HPC/cloud | Proceed with in-place wiring for **that** target; flag "confirm with admin" |
| Local not ready, no GPU access | Complete inspect + mapping + optional in-place wiring; **no** pretend execution |
| User declines checks | Mark `unknown`; still ask HPC vs cloud if they plan to run accelerated workflow |

Do not install Parabricks cluster-wide without explicit user request. Do not store cloud credentials in the repo.
