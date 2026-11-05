# AR controls and supported application paths

Read [AR alpha](ar-compositing.md), [settings and recovery](settings-and-lifetime.md)
and upstream session/context guidance through [the SDK route](../dependencies/ovxr.md).
This reference defines application policy and UI state. It does not supply a
live RenderProduct mutation API or claim support from an unreleased build.

## Keep policies independent

| Application control | Product intent |
| --- | --- |
| Transparent background | Produce alpha while preserving dark opaque objects and fractional edge coverage. |
| Dome background | Choose camera visibility separately from illumination. |
| Refraction as opacity | Declare how transparent materials contribute to alpha. |
| Fractional cutouts | Preserve the requested edge coverage; measure rendering cost. |
| Matte objects | Apply only to appropriately authored receivers. |
| Shadows only | Show the requested receiver contribution without changing unrelated materials. |
| Auto exposure | Persist the chosen exposure policy independently of passthrough. |

Map these intents to the selected renderer's public schema and supported
configuration path. Schema presence or a successful attribute write proves
neither live invalidation nor correct pixels. Keep the OpenXR session/blend
request separate: an alpha checkbox cannot renegotiate an opaque runtime.

## Select an available apply path

For each control, identify a current public setting/API and whether it applies
before product initialization or through a documented live operation. Record
package identities, supported control keys and actual application behavior.
Do not infer capability from a version string, private change, or prior test
against another artifact set.

If a supported live operation exists, execute it on the runtime owner at its
documented boundary through the shared publication coordinator. Validate all
keys/types and expected revision before mutation. Keep every eye consistent;
if partial failure cannot be recovered by a documented operation, report error
or indeterminate state and stop claiming applied success.

If only initialization is supported, retain the requested policy as staged.
Offer explicit reconfiguration only when the public initialization path can
actually express that setting. Context recreation alone does not guarantee
arbitrary eye-product settings will apply. When neither path is supported,
show the control unavailable and explain the capability; do not write arbitrary
attributes, add another render step, or silently restart on each click.

## UI, persistence and completion

An application may implement a correlated settings request with a partial
policy and expected revision. Reply with the canonical full policy, accepted
revision, applied revision, lifecycle lane, state (`staged`, `applied`,
`unavailable`, `error`) and a bounded reason. These are application messages,
not SDK exports. Native acceptance and committed values do not establish
renderer consumption or inspected client pixels.

Group controls by Passthrough, Materials and shadows, and Exposure. Coalesce
updates or keep one in flight, show pending state, and avoid overwriting a new
acknowledgement with older status. Distinguish save failure from apply failure.
After reconnect, query canonical state and reapply only through supported paths;
a timeout does not authorize replaying a mutation blindly.

Preserve deliberate renderer overrides. Migrate application presets explicitly
rather than automatically enabling a global AR bundle. Session blend mode uses
its own lifecycle control.

## Validate the chosen path

- Verify each supported setting in both directions on every affected eye.
- Compare actual RGBA with dark opaque geometry, thin edges, glass/cutout and
  matte-receiver fixtures as relevant; a plain cube cannot prove all effects.
- Require continued head motion, submissions and new client pixels after apply.
- Confirm retained policy after eligible reattachment without another UI edit.
- Report changed renderer/context/service identities for a reconfiguration lane.
- Keep pre-compositor alpha, final projection, desktop and headset evidence
  distinct. Bound diagnostic captures; they are not a steady-state frame hook.

Report live support only for the selected public path and tested combination.
Unsupported live controls remain unavailable even if stored policy can be edited.
