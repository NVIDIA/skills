# jetson-customize-uphy — Run-state JSON Sidecar

Persist Step-5 answers to a JSON sidecar at
`<workspace>/target-platform/<profile-stem>.jetson-customize-uphy.json`
(2-space indent):

```json
{
  "generator": "jetson-customize-uphy",
  "generator_version": "0.0.2",
  "generated_at": "<ISO-8601 UTC>",
  "active_profile": "<profile-stem>.yaml",
  "bsp_version": "<bsp_image.version>",
  "uphy": {
    "config": "uphy0-config-<N>",
    "config_uphy1": "uphy1-config-<M>",
    "lane_assignments": [], "uphy1_lane_assignments": [],
    "carrier_routes_lanes": "yes|no|unknown",
    "warnings": [], "notes": [], "sources": []
  },
  "odmdata_tokens": [],
  "uphy_config_clear_required": false,
  "overlay": {
    "composite_dts_path": "", "composite_dtbo": "",
    "fragments_appended": [
      { "marker": "uphy:<class><id>", "target_path": "/bus@0/<node>",
        "status": "okay|disabled", "reason": "" }
    ],
    "skipped": false
  },
  "commit_shas": { "overlay_tracker": "", "bsp_sources": "" }
}
```

Fields not prompted (UPHY1 on Orin) -> `null`, not absent.

## Idempotency

1. Read prior sidecar; use prior config / config_uphy1 as the **default
   selection** (still ask — never auto-apply); surface prior
   `generated_at`.
2. If new answer matches prior across config + config_uphy1 +
   carrier_routes_lanes byte-for-byte, **skip Steps 6 + 7** and jump to
   Step 8 with `(no change)` headline.
3. Any field differs -> **rewrite the sidecar** (don't merge), then
   proceed.
4. On-disk markers (`# custom-bsp: uphy` in the conf,
   `/* custom-bsp: uphy:<class><id> */` per fragment) are the
   authoritative idempotency mechanism — they survive sidecar deletion.
   JSON is a structured view for tooling.

**Atomic write**: tmp + rename. Never edit in place.
