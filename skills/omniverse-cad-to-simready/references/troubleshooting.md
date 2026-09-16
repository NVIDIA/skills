# CAD to SimReady Troubleshooting

Stage-local symptom/cause/fix detail for the `omniverse-cad-to-simready`
router. Read this reference only when a specific stage is failing or a
validation gate reports a requirement the router's Hard Rules do not spell out
in full. The router (`SKILL.md`) keeps the invariants; this file keeps the
supporting detail.

## Symptom / Cause / Fix Table

| Symptom | Cause | Fix |
|---------|-------|-----|
| Downstream reference reports that cad-to-simready preflight has not prepared a component | `PHYSICAL_AI_REQUIRE_PREFLIGHT=1` is set, but the manifest is missing or the required runtime/service is not `ready` | Run `preflight/scripts/preflight.py`, source the generated env file, or explicitly disable service deployment with `--skip-content-agents` only when Content Agents are out of scope. |
| Workflow stops on `GSP.001` and reports the failure as unclassified | Visual evidence or explicit grasp points were not provided to FET005 | Run upstream `simready-foundation-conform-fet-005-simulate-grasp-physics` only after a vision-capable agent has reviewed the asset, or pass explicit grasp points. Otherwise report the FET005 step as `blocked`, not failed. |
| Validation fails after a meaningful USD artifact already exists | Workflow stopped at the first validation finding | Continue remaining diagnostic gates and mark the result `needs_rerun`. Do not stop at validation findings once a USD artifact has been produced. |
| Property-assignment stage fails with a missing service endpoint | Content Agents service was not deployed before conversion | Run `deploy-content-agents` first. Do not start asset inspection, conversion, validation, conformance, rendering, or packaging before Content Agents readiness when property assignment will run. |
| Material Agent reports that rendering produced `0 images` after unit or profile repair | A FET repair, commonly FET001 unit normalization, was applied before Material Agent and changed the USD layering/scene state consumed by the service | Rerun assignment from the converted/minimum-valid USD: Material Agent first, then Physics Agent, then run `simready-conform-profile` and FET repairs on the latest service-authored USD. |
| Material or Physics Agent local optimized path reports `Permission denied: '/app/.build-resources/scene_optimizer_core/python'` | Local Docker Scene Optimizer bundle permissions prevent the non-root service user from reading the packaged SO runtime | Repair the relevant local container with `docker exec --user root content-material-agent-service chmod -R a+rX /app/.build-resources/scene_optimizer_core` or `docker exec --user root content-physics-agent-service chmod -R a+rX /app/.build-resources/scene_optimizer_core`, then rerun the same optimized agent command. Do not treat the no-optimizer fallback as the root cause for instanced/prototype assets. |
| `RB.MB.001` fails even though the asset has many prims | The profile counts `UsdPhysics.RigidBodyAPI` prims, not visual or collider prims; Physics Agent may author one root rigid body | Route to upstream `simready-foundation-conform-fet-004-simulate-multi-body-physics`. First ensure Physics Agent used composed-topology optimization when applicable, then promote existing component colliders/part roots when the active profile reports FET004/RB.MB.001 and no geometry must be invented. |

## FET Repair Routing Detail

`simready-conform-profile` routes feature repair to upstream SimReady
Foundation FET skills such as `simready-foundation-conform-fet-000-core`,
`simready-foundation-conform-fet-001-minimal`,
`simready-foundation-conform-fet-004-simulate-multi-body-physics`, and
`simready-foundation-conform-fet-005-simulate-grasp-physics` from branch
`main`. Use it only after property assignment when
`property_assignment_intent=run`.

If `simready-validate` reports a repairable requirement after the first
conformance pass, feed the structured requirement IDs back into the
`simready-conform-profile` reference before writing the final result.

- `GSP.001` is owned by upstream
  `simready-foundation-conform-fet-005-simulate-grasp-physics`; run that skill
  when a vision-capable agent can inspect visual evidence or explicit grasp
  points were provided, otherwise record the FET005 step as blocked by missing
  vision/points instead of treating it as an optional preview task.
- For `RB.MB.001`, route the failure to upstream
  `simready-foundation-conform-fet-004-simulate-multi-body-physics`. Do not
  assume multiple visual prims are multiple rigid bodies; inspect
  `UsdPhysics.RigidBodyAPI` applications. When the Physics Agent report shows
  composed topology optimization or the USD has existing component
  colliders/part roots and the profile validator reports FET004/RB.MB.001,
  FET004 should promote those existing components into rigid bodies without
  creating geometry. Do not mark the gate not applicable until after
  confirming there are fewer than two reusable body candidates.
- Upstream `simready-foundation-conform-fet-005-simulate-grasp-physics` needs
  visual review or explicit grasp points before it can author a meaningful
  grasp vector.
