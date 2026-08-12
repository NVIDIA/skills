# SimReady FET004 Multi-Body Physics Local Helper

## Upstream Skill

Source of truth:

```text
https://github.com/NVIDIA/simready-foundation/blob/v2026.04.1/skills/simready-foundation-conform-fet-004-simulate-multi-body-physics/SKILL.md
```

Use an authenticated local checkout at
`$SIMREADY_FOUNDATION_ROOT/skills/simready-foundation-conform-fet-004-simulate-multi-body-physics/SKILL.md`
or
`$OMNIVERSE_CAD_TO_SIMREADY_UPSTREAM_ROOT/simready-foundation/skills/simready-foundation-conform-fet-004-simulate-multi-body-physics/SKILL.md`
when browser access is unavailable.

Do not copy FET004 requirement summaries or repair policy into this repo.

## Local Helper

This directory only keeps a legacy Omniverse CAD-to-SimReady helper script for promoting
existing component rigid bodies and writing JSON reports:

```bash
python3 scripts/run.py <usd-asset> \
  --output-dir <output-root>/conform/fet004 \
  --profile Prop-Robotics-Neutral \
  --profile-version 1.0.0 \
  --report <output-root>/fet004-multibody.json
```

Read the upstream Foundation skill before using the helper. Treat the upstream
skill as authoritative when its instructions differ from this local wrapper.

## Next Step

After multibody conformance, rerun `simready-validate`. If the asset has only
one mesh component or one `GeomSubset` component, let `simready-validate` apply
the local non-blocking `RB.MB.001` policy instead of inventing geometry.
