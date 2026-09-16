# Showcase Workflow

Read references in stage order; do not load them all up front.

| Stage | Read | Adapter |
|---|---|---|
| Dependency readiness | nothing | `scripts/doctor.py dependencies` |
| Host inspection | nothing | `scripts/doctor.py host` |
| Isaac Sim absent | `references/isaac-sim-installation/README.md` | none, by design |
| Robot identity | `references/change-fleet-composition/README.md` | `.../scripts/run.py` |
| Map configuration | `references/change-map/README.md` | `.../scripts/run.py` |
| Cloud startup | `references/bring-up-cloud-stack/README.md` | `.../scripts/run.py` |
| Stage control | `references/isaac-sim-remote/README.md` | `.../scripts/run.py` |

Each reference is a parent-owned boundary contract. The behavior belongs to the
upstream skill it fronts; read that skill's runtime `SKILL.md`, at the path the
manifest reports, before invoking it.

The parent keeps everything that spans components: Isaac Sim lifecycle, the
readiness contract, run state and interruption handling, acceptance, motion
evidence, and targeted stop. Those are in `SKILL.md` and `references/troubleshooting.md`.
