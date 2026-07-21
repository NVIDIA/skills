# Hybrid Env Layout Rules

Load when the requested env combines a scene from one env with a robot/policy from another, or when G1 footprint, table height, USD scale, or catalog assets matter.

## Hybrid Envs (Scene-of-A + Robot-of-B)

When the chosen scene source and robot come from different envs, this is a
**hybrid**. Do not pick the construction approach yourself — present these
options and let the user choose before editing:

- **Robot integration** (ask): (a) the robot-owner's Arena embodiment
  (e.g. `HumanoidEnvironmentBase` + the registry embodiment for G1) — brings the
  WBC action space + head camera the robot's policy stack drives, **required to
  run that stack's policy**; or (b) a raw IsaacLab `ArticulationCfg` in the scene
  cfg — no WBC / head camera, so it can't run a WBC policy.
- **Scene asset pattern** (ask only if scene source and robot-owner differ):
  keep the scene source's pattern — inline `InteractiveSceneCfg` + `ConfigAsset`
  - `make_<env>_scene_assets()` (scissor) or the `@register_asset` registry
  (locomanip). Do not mix the two.

Once the user has chosen, wire it up per the Files to Create table (env extends
the robot-owner's base — `HumanoidEnvironmentBase` for G1), plus these
hybrid-specific rules:

- For G1, the embodiment provides its own head camera via `G1EmbodimentBase.get_scene_cfg()`. Do not add cameras to the forked `InteractiveSceneCfg`.
- Keep the scene source's `ground = AssetBaseCfg(GroundPlaneCfg, ...)` field. Without a ground plane the G1 falls into the void. Pair ground z with the WBC base-height command (`apply_wbc_default_base_height`) per G1 vertical setup below.
- Static destination assets (trays, fixtures) that use `SCISSOR_TRAY_USD` must spawn `kinematic_enabled=True, disable_gravity=True`. Dynamic spawning settles the visual rim into the tabletop.

## Robot Reach

The env class's `embodiment.set_initial_pose(...)` sets where the robot stands; the assets file's `init_state.pos` sets where props start. Position the work zone within reach:

| Robot | Standing world x | Work-zone x |
|---|---|---|
| SO-ARM 101 | zero-translation tabletop mount | table center to about 30 cm forward |
| Unitree G1 (forked table) | small `−x` just behind a forward-offset table (e.g. `≈−0.4`), identity rotation, facing the table | tabletop within reach in front |
| Unitree G1 (locomanip room) | well back on open floor (`x≈−1.0` or further) | `-0.2 … 0.2` via walking policy |

When props default outside reach, move the table (with its props/destinations), not just the props.

**Locomanip room vs. table-based envs:** The G1 locomanip room env (`locomanip_tray_pick_and_place`) stands the G1 far back on open floor and lets the policy walk up to props. A forked **scissor table** is centered at `x≈0`, so standing the G1 at that same `x` puts it *inside* the table footprint and the WBC topples. Two layouts avoid that — pick one; both are valid:

- **Table-forward (compact):** offset the table forward in `+x` (props/destinations move with it) so its near edge clears the robot, and stand the G1 just behind at a small **negative x**, facing the table with **identity rotation `(1,0,0,0)`**. This is the compact layout a scissor-table G1 hybrid uses — the robot is close to the work surface, no long walk needed.
- **Walk-up (open floor):** stand the G1 well back (`x≈-1.0` or further, open floor) and let the loco-manip policy walk up — the locomanip room style.

Do not stand the robot far out on the **same (+x) side as the table** or apply a 180° yaw to face back toward it; seat it on the opposite (−x) side facing the table.

Verify clearance from live poses (`GET /object?name=robot`/`table`), not the bbox — see the Phase 1 probe method.

**Footprint clearance — a free-standing robot must not stand inside the support
surface (a horizontal decision, separate from the vertical height setup).** The
G1 standing band above suits *locomanip room* scenes (open floor); a forked
**scissor table is centered at `x≈0`** (`SCISSOR_TABLE_USD` spans `x ∈ [-0.40,
+0.40]`), so that band puts the G1's torso/thighs *inside* the table and **the WBC
topples on reset — a body–table collision, not a height problem.** The G1 occupies
roughly `x ∈ [root-0.08, root+0.42]` (arms forward), so keep the table's near edge
in front of that. Two options (footprints must not overlap):

- **Offset the forked table forward (`+x`)** — for example, put the table roughly 45 cm forward so its near edge clears the robot, stand the G1 roughly 40 cm behind the origin, and move props/destinations with the table.
- **Stand the G1 well back (`x≈-1.0` or further)** on open floor and let the loco-manipulation policy walk up — how `locomanip_tray_pick_and_place` is laid out, and why a too-close G1 becomes stable once moved away from the table.

Verify clearance from the live poses (`GET /object?name=robot`/`table`), not the bbox — see the Phase 1 probe for the method.

### G1 vertical setup

Pair the ground z with the WBC base-height command: a ground at `z=-X` pairs with
`apply_wbc_default_base_height(embodiment, base_height_m=X)`, called in `get_env`
(WBC default 75 cm). **`X` must be at least the standing foot depth**: a standing
G1 with an 80 cm base height has its waist near the origin and feet roughly 79 cm
below it, so a shallower ground penetrates the feet and topples the WBC on reset —
use matching 80 cm ground depth and base-height values. Put the tabletop **about
1 ft below the waist** —
SO-ARM-derived tables default to chest height (`z≈0.238`), too high. Pick
`scale_z` + `pos.z` so the tabletop hits the target while the legs rest on the
ground: `pos.z ± half_height = tabletop_z` / ground depth. For `SCISSOR_TABLE_USD`
at the low tabletop target, one known-good source setting is
`spawn.scale=(70/100, 70/100, 547/1000)`, `init_state.pos.z=-55/100` (50 cm tall,
top around 30 cm below waist / legs at the 80 cm ground depth — legs clipping the
floor is cosmetic). Place props by rendered bbox, not cfg root: cube/tool bbox bottoms
should sit on the tabletop, and `SCISSOR_TRAY_USD` trays must have bbox bottoms
within 0-2 mm of the tabletop with no visible air gap.

Resizing is **source-only** (+ relaunch): you can't rescale a support surface
live — its cooked collision mesh keeps the old size and props fall through. To
preview a height in edit mode, *translate* the kinematic body down instead (see
[[i4h-workflow-scene-edit]] — "live-move embedded rigid body").

## Adding New USD Assets

When the task needs a prop the workflow doesn't already use:

- Prefer the healthcare catalog: <https://github.com/isaac-for-healthcare/i4h-asset-catalog/blob/main/catalog.md>; fall back to generic Isaac Sim / Isaac Lab assets only when no healthcare USD fits.
- Discover the exact USD path by listing the public S3 bucket:

  ```bash
  curl -s 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/?list-type=2&prefix=Assets/Isaac/Healthcare/0.5.0/132c82d/Props/'
  ```

- Verify the USD's authored scale via the bridge `/object` bbox before picking a `scale=` tuple. Catalog USDs ship at varying unit lengths; for example, scissors need a millimeter-scale factor while surgical tweezers use unit scale.
- For static destination assets, also follow the kinematic + gravity-off rule above, pin `init_state.pos.z` so the bbox bottom is 0-2 mm above the tabletop with no visible air gap, and reuse `_asset_world_position` for success checks (IsaacLab classifies `AssetBaseCfg` prims as XformPrim regardless of `rigid_props`; `asset.data.root_pos_w` is absent).
