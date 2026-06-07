"""Minimal ovrtx step test — run to isolate crash cause.

Tests three scenarios in order:
  1. Create Renderer (no scene)  →  try step  →  should return None, not crash
  2. Load EMPTY inline scene     →  try step  →  should return frame or None
  3. Load SUBLAYER scene         →  try step  →  needs scene file to exist

Run:
    python test_renderer.py
    python test_renderer.py --no-sublayer   # skip scenario 3
"""
import os, sys, time, argparse

os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

EMPTY_USDA = """\
#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World" {
    def Cube "Cube" {
        double size = 100.0
    }
    def DomeLight "DomeLight" {
        float inputs:intensity = 1000.0
    }
}

def Camera "OVCamera" (
    prepend apiSchemas = ["CameraAPI"]
) {
    float2 clippingRange = (1.0, 1000000.0)
    float focalLength = 50.0
    float horizontalAperture = 36.0
    float verticalAperture = 24.0
    double3 xformOp:translate = (0.0, 200.0, 500.0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

def Scope "Render" {
    def RenderSettings "OVSettings" {
        rel products = [</Render/OVServer>]
        token aspectRatioConformPolicy = "expandAperture"
        int2 resolution = (640, 360)
    }

    def RenderProduct "OVServer" {
        rel camera = </OVCamera>
        token productType = "raster"
        int2 resolution = (640, 360)
        rel orderedVars = [</Render/Vars/LdrColor>]
    }

    def Scope "Vars" {
        def RenderVar "LdrColor" {
            token dataType = "color4f"
            custom string driver:parameters:aov:name = "LdrColor"
            token sourceName = "LdrColor"
            token sourceType = "raw"
        }
    }
}
"""


def make_cfg():
    import ovrtx
    try:
        return ovrtx.RendererConfig(sync_mode=True)
    except TypeError:
        try:
            return ovrtx.RendererConfig()
        except Exception:
            return None


def try_step(renderer, rp_path, label):
    print(f"  step({rp_path!r}) ...", end=" ", flush=True)
    try:
        try:
            frame = renderer.step(rp_path, 1.0/30)
        except TypeError:
            frame = renderer.step(rp_path)
        print(f"OK  frame={type(frame).__name__ if frame is not None else None}")
        return True
    except Exception as exc:
        print(f"EXCEPTION: {exc}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sublayer", action="store_true")
    args = ap.parse_args()

    print("\n=== Scenario 1: Renderer only, no scene ===")
    import ovrtx
    print(f"  ovrtx version: {getattr(ovrtx, '__version__', 'unknown')}")
    cfg = make_cfg()
    r = ovrtx.Renderer(config=cfg) if cfg else ovrtx.Renderer()
    print("  Renderer created OK")
    # step without any scene loaded — should be a no-op / return None
    try_step(r, "/Render/OVServer", "no-scene")
    del r
    print("  Renderer deleted OK")

    print("\n=== Scenario 2: Empty inline USDA (no sublayer) ===")
    r = ovrtx.Renderer(config=make_cfg()) if make_cfg() else ovrtx.Renderer()
    try:
        r.open_usd_from_string(EMPTY_USDA)
        print("  open_usd_from_string OK")
    except Exception as exc:
        print(f"  open_usd_from_string FAILED: {exc}")
        del r
        return
    time.sleep(0.5)  # give population thread a chance to finish
    ok = try_step(r, "/Render/OVServer", "empty-scene")
    del r
    if not ok:
        print("  CRASH in empty inline USDA — platform-level issue, not sublayer related")
        return

    if args.no_sublayer:
        print("\nSkipping scenario 3 (--no-sublayer)")
        return

    scene_file = "assets/samples/scene.usda"
    if not os.path.exists(scene_file):
        print(f"\nSkipping scenario 3: {scene_file!r} not found")
        return

    print("\n=== Scenario 3: Sublayer user scene ===")
    abs_path = os.path.abspath(scene_file).replace("\\", "/")
    sublayer_usda = EMPTY_USDA.replace(
        'defaultPrim = "World"',
        f'defaultPrim = "World"\n    subLayers = [\n        @{abs_path}@\n    ]'
    )
    r = ovrtx.Renderer(config=make_cfg()) if make_cfg() else ovrtx.Renderer()
    try:
        r.open_usd_from_string(sublayer_usda)
        print(f"  open_usd_from_string with sublayer OK")
    except Exception as exc:
        print(f"  open_usd_from_string FAILED: {exc}")
        del r
        return
    time.sleep(0.5)
    try_step(r, "/Render/OVServer", "sublayer-scene")
    del r

    print("\n=== All scenarios completed ===")


if __name__ == "__main__":
    main()
