# Existing Scene and Asset Facts

Use this reference before measuring, rescaling, or repositioning an asset that already appears in a maintained scene. These are warm-start facts copied from the current owning sources; the Python source remains authoritative if this reference and code ever disagree.

## Reuse order

1. Find the closest scene below and inspect its listed owner.
2. Reuse the existing USD identity, authored scale, physics role, support height, and embodiment convention.
3. Use `scripts/live_scene_edit.py add-known-asset` when the asset has an executable preset in `arena/i4h_arena/assets/authoring_catalog.py`.
4. Inspect live world bounds after insertion. Do not search for a scale already recorded here.
5. Use visual-language review only to refine the resulting scene composition, never to infer the initial physical scale from scratch.

The executable preset catalog currently contains measured metric bounds for `surgical_table`, `scissors`, `tweezers`, `surgical_tray`, and `g1`. Its 20% bounds guard detects a broken reference, unexpected root transform, or unit mismatch. Robot presets also carry the registered embodiment, live-to-runtime root mapping, action contract, control rate, attached cameras, and camera aliases returned by `scripts/authoring_info.py`. Assets listed only in the tables below retain their proven source-authored scale and placement until measured bounds are added to the catalog.

## Scene index

| Workflow scene | Primary owner | Reusable scene facts |
|---|---|---|
| `soarm_scissors` | `arena/i4h_arena/assets/soarm_scissors.py` | Ground z `-1.05`; scissor table at `(0.1, 0, 0)`, z-rotation `90°`, scale `(0.7, 0.7, 0.52)`; tabletop world z is approximately `0.238`; SO-ARM pose belongs to its embodiment, not the Scene cfg. |
| `g1_tray` | `arena/i4h_arena/scenes/_locomanip.py`, `g1_tray.py`; assets in `arena/i4h_arena/assets/_locomanip.py` | Rheo `pre_op` background at `(4, 0, -0.8)`; tray at `(-1.15, -1.6, -0.08)` with z-rotation `90°`; cart at `(0.35, -1.65, -0.7875)`; G1 at `(-0.5, -1.62, 0)` facing the work area. Registered assets and G1 use scale `1`. |
| `g1_cart` | `arena/i4h_arena/scenes/_locomanip.py`, `g1_cart.py`; assets in `arena/i4h_arena/assets/_locomanip.py` | Same Rheo background and destination cart as `g1_tray`; active cart prop at `(0.35, -1.65, 0.10)` with z-rotation `90°`; G1 at `(-0.4, -1.62, 0)`. Registered assets and G1 use scale `1`. |
| `g1_trocar` | `arena/i4h_arena/scenes/g1_trocar.py`; assets in `arena/i4h_arena/assets/g1_trocar.py` | LightWheel room and all registered props use scale `1`; `trocar_1` at `(-1.60202, 1.91362, 0.87183)`, `trocar_2` at `(-1.50635, 1.90997, 0.8631)`, tray at `(-1.54919, 2.03365, 0.84554)`; preserve the exact source quaternions. Uses the G1 Dex embodiment and front/left-wrist/right-wrist cameras. |
| `panda_phantom` | `arena/i4h_arena/assets/panda_phantom.py` | Ground z `-0.84`; covered table at `(0.4804, 0.02017, -0.84415)` with z-rotation `-90°`; phantom at `(0.6, 0, 0.09)` with z-rotation `180°`, scale `1`, rigid mass `1000 kg`; goal frame offset `(0, -0.25, 0.75)`. |
| `psm_reach` | `arena/i4h_arena/assets/_surgical.py` | General surgical table at `(0, 0, -0.457)`, scale `1`; ground z `-0.95`; reach marker is a `0.015 m` radius sphere at `(0.02, 0, 0.055)`. |
| `dual_psm_reach` | `arena/i4h_arena/assets/_surgical.py` | Reuses the general surgical table and reach workspace from `psm_reach`; the dual PSM embodiment owns both robots and the scene exposes two command targets. |
| `star_reach` | `arena/i4h_arena/assets/_surgical.py` | Uses `SeattleLabTable`, not the general surgical table; table at `(0.55, 0, 0)`, z-rotation `90°`, scale `1`; reuses the `0.015 m` reach marker. |
| `psm_block` | `arena/i4h_arena/assets/_surgical.py` | Reuses the general surgical table; block at `(0, 0, 0.025)`, scale `(0.011, 0.011, 0.011)`, rigid with gravity. |
| `psm_needle` | `arena/i4h_arena/assets/_surgical.py` | Reuses the general surgical table; SDF needle at `(0, 0, 0.015)`, scale `(0.4, 0.4, 0.4)`, rigid with gravity. |
| `psm_needle_organs` | `arena/i4h_arena/assets/_surgical.py` | Full OR organ scene at `(0.25, -0.14, -0.85)`, z-rotation `90°`, scale `(0.01, 0.01, 0.01)`; non-SDF needle at `(0, 0, 0.015)`, scale `(0.4, 0.4, 0.4)`, gravity disabled because the organ USD lacks reliable support collision. |

## Reusable asset details

| Asset | Source constant or type | Proven scale | Physics and placement note |
|---|---|---|---|
| Scissor table | `SCISSOR_TABLE_USD` | `(0.7, 0.7, 0.52)` | Static support; measured size is in the executable catalog. |
| Surgical scissors | `SCISSORS_USD` | `(0.006, 0.0065, 0.012)` | Rigid, `0.15 kg`; place just above the table support surface. |
| Surgical tweezers | `SURGICAL_TWEEZERS_USD` | `(1, 1, 1)` | The executable preset uses rigid mass `0.05 kg`; measured size is in the catalog. |
| Small surgical tray | `SCISSOR_TRAY_USD` | `(0.7, 0.7, 0.18)` | The executable preset is static; the `soarm_scissors` variant authors `5 kg`. |
| Rheo tray with lid | `TRAY_USD` | `(1, 1, 1)` | Articulated, default mass `0.1 kg`; distinct from `SCISSOR_TRAY_USD`. |
| Rheo cart | `CART_USD` | `(1, 1, 1)` | Articulated; preserve its scene-specific support height. |
| G1 WBC robot | `UNITREE_G1_29DOF_USD` | `(1, 1, 1)` | Articulation; live root `/Robot` maps to registered `g1_wbc_joint`, manifest embodiment `g1`, 50-DoF joint-position control at 30 Hz, and runtime robot name `robot`. The live preset creates `Robot/Asset/head_link/RobotHeadCam`; the registered embodiment exposes `robot_head_cam` through alias `head`. |
| General surgical table | `TABLE_USD` | `(1, 1, 1)` | Static support used by PSM reach/lift scenes. |
| STAR table | `SeattleLabTable` | `(1, 1, 1)` | Static support used only by `star_reach`. |
| Lift block | `BLOCK_USD` | `(0.011, 0.011, 0.011)` | Rigid with gravity. |
| SDF needle | `NEEDLE_SDF_USD` | `(0.4, 0.4, 0.4)` | Rigid with gravity and tuned solver properties. |
| Organ-scene needle | `NEEDLE_USD` | `(0.4, 0.4, 0.4)` | Rigid with gravity disabled in the organ scene. |
| Full OR organs | `ORGANS_USD` | `(0.01, 0.01, 0.01)` | Whole authored room/organ context, not a meter-scale standalone organ prop. |
| Covered ultrasound table | `TABLE_WITH_COVER_USD` | `(1, 1, 1)` | Reuse the source pose before adjusting composition. |
| Abdominal phantom | `PHANTOM_USD` | `(1, 1, 1)` | Rigid, `1000 kg`; preserve its registered frames and goal transforms. |

## Quaternion convention when baking

Arena scene sources carry quaternions as `(x, y, z, w)`, matching `authoring_catalog` `rotation_deg`, `live_scene_edit.py export-scene` `rotation_xyzw`, and `isaaclab_arena.utils.pose.Pose`. Write an exported live rotation straight through; for a catalog yaw use `(0, 0, sin(yaw/2), cos(yaw/2))`. This applies to `init_state.rot` on `AssetBaseCfg`/`RigidObjectCfg` and to `TiledCameraCfg.OffsetCfg.rot` in a `ConfigAsset`-wrapped `InteractiveSceneCfg`; `arena/i4h_arena/assets/soarm_scissors.py` shows the same form for the table, scissors, and tray USDs. Do not "convert to IsaacLab's documented `(w, x, y, z)`" while baking: on these assets that reads as a 90° roll, which stands instruments on end and aims a room camera at the sky — both of which still pass lint.

Confirm the convention the same way after every bake: relaunch `./run.sh <workflow> --live`, then compare each prim's world bounds and each camera capture against the live session they came from.

## Visual-language refinement

Start from the known preset or closest scene, render the room view and task camera, and compare object support, mutual proportions, visibility, and robot reachability. Keep automatic refinement bounded to a scene-local override of at most ±20% per axis unless the user requests a different physical size or measured geometry proves the preset wrong. Recheck collision, support height, camera visibility, and task reach after each refinement.

Do not update the shared preset from one image. Promote a refinement into `authoring_catalog.py` only when world bounds or trusted physical dimensions establish a better canonical value across consumers; update the catalog test and every affected scene together.
