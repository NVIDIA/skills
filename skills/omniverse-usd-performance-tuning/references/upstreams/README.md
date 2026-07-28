# Upstream Source-of-Truth References

Pointers to the upstream repositories and prebuilt packages this skill delegates
to instead of reimplementing. Operation mechanics, parameters, defaults, and
package resolution live upstream; this skill owns only the digital twin workflow
routing, runtime setup, validation scope, output policy, and reporting that wrap
them.

When a file here names a tool, prefer the upstream URL it records for the most
current version — the local notes are a snapshot and a resolution recipe, not a
copy of the upstream docs.

## Contents

- [`usd-optimize.md`](usd-optimize.md) — Usd Optimize operation mechanics and
  prebuilt-package resolution (upstream
  [usd-optimize](https://github.com/NVIDIA-Omniverse/usd-optimize/)). Resolve
  per-operation guides through `$USD_OPTIMIZE_ROOT` with the version-tolerant
  lookup stated in `usd-optimize.md` (`docs/operations/<key>.rst` on 1.1.x
  packages, `.agents/operations/<key>.md` on 1.0.x) rather than duplicating
  them in this repo.
