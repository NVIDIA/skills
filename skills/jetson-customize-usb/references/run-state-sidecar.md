# jetson-customize-usb — Run-state JSON sidecar

Path: `<workspace>/target-platform/<profile-stem>.jetson-customize-usb.json`

```json
{
  "generator": "jetson-customize-usb",
  "generator_version": "0.1.0",
  "generated_at": "<ISO-8601 UTC>",
  "active_profile": "<profile-stem>.yaml",
  "bsp_version": "<bsp_image.version>",
  "ports": [
    {"id": "usb2-3", "action": "disable", "lane_phandle": 347, "host_phys_index": 3, "stock_status": "okay", "stock_mode": "host", "reason": "wired to on-carrier USB3 hub"},
    {"id": "usb3-2", "action": "disable", "lane_phandle": 349, "host_phys_index": 6, "stock_status": "okay", "cascaded_from": "usb2-3", "reason": "nvidia,usb2-companion = <usb2-3 phandle>"}
  ],
  "host_phys_keep": ["usb2-0", "usb2-1", "usb2-2", "usb3-0", "usb3-1"],
  "host_phys_phandles_keep": [343, 345, 346, 348, 344],
  "companion_graph": {"usb2-3": ["usb3-2"]},
  "overlay": {
    "composite_dts_path": "...tegra-soc-carrier-id-sku-module-id-xxxx-custom.dts",
    "fragments_appended": ["usb:padctl", "usb:xhci", "usb:xudc"],
    "post_merge_invariants": "verified"
  },
  "warnings": [], "notes": [], "sources": [],
  "commit_shas": {"overlay_tracker": "<short SHA>", "bsp_sources": "<short SHA>"}
}
```

Atomic write + idempotency contract: same as `jetson-customize-uphy`
Step 5.
