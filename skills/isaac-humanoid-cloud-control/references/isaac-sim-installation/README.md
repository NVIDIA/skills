# Installation handoff

There is no script adapter for installation. When Isaac Sim is absent or a
version change is explicitly requested, read the runtime `SKILL.md` recorded at
`skills.isaac-sim-installation.skill_md` in the dependency manifest and follow
it directly. Preserve its method, compatibility, EULA, and execution gates.
Preselect standalone for this browser-streamed workflow. Stop installation
before launch, then return to the humanoid bootstrap/readiness workflow.

A missing or unready dependency is a blocker; never search for another installer
or reconstruct its implementation. The official viewer source is a separate
dependency with the exact runtime tag and authorization described in bootstrap.
