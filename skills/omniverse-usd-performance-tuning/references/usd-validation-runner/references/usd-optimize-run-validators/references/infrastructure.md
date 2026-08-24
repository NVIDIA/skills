# Usd Optimize Validator Infrastructure - Upstream Handoff

This local reference preserves the digitaltwin workflow milestone. Scene
Optimizer mechanics for this step are owned by upstream `usd-optimize`.

- Public repository: [https://github.com/NVIDIA-Omniverse/usd-optimize/](https://github.com/NVIDIA-Omniverse/usd-optimize/)
- Package path: `.agents/skills/new-validator/SKILL.md` on 1.1.x, `.agents/skills/validators/SKILL.md` on 1.0.x
- Upstream web URL: [https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/skills/new-validator/SKILL.md](https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/.agents/skills/new-validator/SKILL.md)

Resolve the upstream guide without cloning the source repo. Upstream renamed this
skill in 1.1.0, so take the first path that exists:

1. `$USD_OPTIMIZE_ROOT/.agents/skills/new-validator/SKILL.md` (1.1.x)
2. `$USD_OPTIMIZE_ROOT/.agents/skills/validators/SKILL.md` (1.0.x)

If no package root is available, download and extract the prebuilt Usd Optimize release package (current asset name + download: `references/upstreams/usd-optimize.md`) (direct
archive URLs are in `references/upstreams/usd-optimize.md`), or use the package
path/URL supplied by the user. If the user supplies an extracted
package root directly, resolve this same package path under that root. If
GitHub raw fetch is available, the web URL above is acceptable for docs-only
reads. Do not clone the source repo just to read upstream SO guidance.

## Local Responsibilities

- Local validation scope, phase-aware subsets, and expensive-check gates remain in `usd-validation-runner/README.md`.
- Setup/install references own runtime selection and `setup-preflight.json` writer behavior.
