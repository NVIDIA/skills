"""Builds inline USDA roots that sublayer user scenes.

Never modifies user USD files. Camera, RenderProduct, RenderVar, and
RenderSettings are injected into a root layer that sublayers the user file.

Refer to omniverse-realtime-viewer → references/stage-loading/ for the
full composition and camera-injection contract.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

VIEWER_CAMERA_PATH    = "/OVCamera"
RENDER_PRODUCT_PATH   = "/Render/OVServer"
RENDER_SETTINGS_PATH  = "/Render/OVSettings"


def _render_block(width: int, height: int) -> str:
    """Returns the Render scope block shared by both USDA variants."""
    return f"""def Camera "OVCamera" (
    prepend apiSchemas = ["CameraAPI"]
) {{
    float2 clippingRange = (1.0, 10000000.0)
    float focalLength = 50.0
    float horizontalAperture = 36.0
    float verticalAperture = 24.0
    double3 xformOp:translate = (0.0, 200.0, 500.0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}}

def Scope "Render" {{
    def RenderSettings "OVSettings" {{
        rel products = [</Render/OVServer>]
        token aspectRatioConformPolicy = "expandAperture"
        int2 resolution = ({width}, {height})
    }}

    def RenderProduct "OVServer" {{
        rel camera = </OVCamera>
        token productType = "raster"
        int2 resolution = ({width}, {height})
        rel orderedVars = [</Render/Vars/LdrColor>]
    }}

    def Scope "Vars" {{
        def RenderVar "LdrColor" {{
            token dataType = "color4f"
            custom string driver:parameters:aov:name = "LdrColor"
            token sourceName = "LdrColor"
            token sourceType = "raw"
        }}
    }}
}}
"""


def _inline_usda_with_sublayer(scene_path: str, width: int, height: int) -> str:
    abs_path = os.path.abspath(scene_path).replace("\\", "/")
    render = _render_block(width, height)
    return f"""#usda 1.0
(
    subLayers = [
        @{abs_path}@
    ]
)

{render}"""


def _inline_usda_empty_scene(width: int, height: int) -> str:
    render = _render_block(width, height)
    return f"""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World" {{
    def Cube "Cube" {{
        double size = 100.0
        double3 xformOp:translate = (0.0, 50.0, 0.0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def DomeLight "DomeLight" {{
        float inputs:intensity = 1000.0
        color3f inputs:color = (1.0, 1.0, 1.0)
    }}
}}

{render}"""


class SceneLoader:
    def __init__(self, width: int, height: int, asset_root: str = "") -> None:
        self.width = width
        self.height = height
        self.asset_root = asset_root

    def resolve(self, url: str) -> str:
        if url.startswith(("omniverse://", "http://", "https://", "/")):
            return url
        if self.asset_root:
            return os.path.join(self.asset_root, url)
        return url

    def build_usda(self, scene_url: str) -> tuple[str, str, str]:
        """Return (usda_string, render_product_path, camera_path)."""
        if scene_url:
            resolved = self.resolve(scene_url)
            usda = _inline_usda_with_sublayer(resolved, self.width, self.height)
            log.info("SceneLoader: building inline USDA for %s", resolved)
        else:
            usda = _inline_usda_empty_scene(self.width, self.height)
            log.info("SceneLoader: using empty default scene")
        return usda, RENDER_PRODUCT_PATH, VIEWER_CAMERA_PATH
