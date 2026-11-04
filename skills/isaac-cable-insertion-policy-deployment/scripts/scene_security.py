# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trust boundary for NVIDIA USD assets containing an executable ScriptNode.

The scene and embedded script hashes below match the bytes reviewed and tested
on 2026-10-12. A changed upstream scene requires deliberate code review and a
new pin, never automatic acceptance. Asset URLs must stay in NVIDIA's asset
buckets; local USD references are constructed only from this session's cache.
"""

import hashlib
from pathlib import Path
import tempfile
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SCENE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
    "Isaac/6.1/Isaac/Samples/ROS2/Scenario/flexiv_displayport_insertion.usd"
)
SCENE_SHA256 = "af2965dfe552d53b6e68c6b9a0f9203ac8a42dd65f3282f62be617cbe6efb347"
SCRIPT_SHA256 = "8fc342603633472453e6c80bf69c91f75a1b4bc7ed15dd6c9ac68e4e25a09658"
OSC_SHA256 = "c08572eb65fc5a209dea51021184388e917137bb6cf1961743208657b683e990"
COMBINED_SCRIPT_SHA256 = "70058dfa5f3ca4fec65e57dc535dda8a880aac8224ac590ca9addd4219dd4a67"
SCRIPT_PATH = "/World/ROS/DirectRosPhysicsGraph/DirectRosIo"
ASSET_HOSTS = {
    "omniverse-content-production.s3-us-west-2.amazonaws.com",
    # Historical saved checkpoint assets only; release-r1 defaults use production.
    "omniverse-content-staging.s3-us-west-2.amazonaws.com",
}


def validate_asset_url(url):
    """Allow HTTPS NVIDIA asset URLs only, without credentials or redirection."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ASSET_HOSTS
        or parsed.netloc != parsed.hostname
        or not parsed.path.startswith("/Assets/Isaac/")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query or parsed.fragment
        or any(part in {".", ".."} for part in parsed.path.split("/"))
        or "%" in parsed.path or "\\" in parsed.path
    ):
        raise ValueError(f"Unsupported asset URL: {url}")
    return url


class NoRedirect(HTTPRedirectHandler):
    """Never let a trusted asset URL redirect into a different trust boundary."""

    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise ValueError("Asset download redirect rejected; review the upstream URL")


def check_asset_endpoint(url):
    """Require a direct response before the USD resolver reads an asset URL."""
    validate_asset_url(url)
    request = Request(url, method="HEAD")
    # The URL is HTTPS on an allowlisted NVIDIA bucket, and redirects are disabled.
    with build_opener(NoRedirect()).open(request, timeout=60) as response:  # nosec B310
        if response.status != 200 or response.url != url:
            raise ValueError(f"Asset endpoint did not respond directly: {url}")
    return url


def fetch_scene(cache_dir, url=SCENE_URL):
    """Download a bounded asset file and verify its reviewed SHA256 before use."""
    if url != SCENE_URL:
        raise ValueError("Only the reviewed, hash-pinned NVIDIA tutorial scene is supported")
    validate_asset_url(url)
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "reviewed-displayport-scene.usd"
    if target.is_symlink():
        raise ValueError("Scene cache entry must not be a symbolic link")
    if target.exists():
        payload = target.read_bytes()
    else:
        # HTTPS only and redirects disabled; certificate verification is default.
        with build_opener(NoRedirect()).open(url, timeout=60) as response:  # nosec B310
            payload = response.read(2_000_001)
        if len(payload) > 2_000_000:
            raise ValueError("Scene exceeds the reviewed asset size limit")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != SCENE_SHA256:
        raise ValueError(f"Scene SHA256 mismatch: expected {SCENE_SHA256}, got {actual}")
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=cache_dir, prefix="scene-", delete=False) as output:
            output.write(payload)
            temporary = Path(output.name)
        temporary.replace(target)
    return target


def local_scene_layer(scene_path, cache_dir):
    """Resolve reviewed relative references while preserving the original bytes."""
    from pxr import Sdf

    # Revalidate at the last boundary before USD parsing.
    payload = Path(scene_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != SCENE_SHA256:
        raise ValueError("Scene changed before USD parsing")
    source = Sdf.Layer.FindOrOpen(str(scene_path))
    layer = Sdf.Layer.CreateAnonymous("reviewed-displayport.usda")
    layer.TransferContent(source)

    def visit(prim):
        references = list(prim.referenceList.prependedItems)
        if references:
            prim.referenceList.prependedItems = [
                Sdf.Reference(
                    validate_asset_url(urljoin(SCENE_URL, reference.assetPath)),
                    reference.primPath, reference.layerOffset, reference.customData,
                )
                for reference in references
            ]
        for child in prim.nameChildren.values():
            visit(child)

    for prim in layer.rootPrims:
        visit(prim)
    target = Path(cache_dir).resolve() / "resolved-displayport-scene.usda"
    if target.is_symlink():
        raise ValueError("Resolved scene cache entry must not be a symbolic link")
    if not layer.Export(str(target)):
        raise RuntimeError("Could not save the reference-resolved reviewed scene")
    return target


def verify_script(stage):
    """Check the executable payload and reject additional scene ScriptNodes."""
    script_nodes = [
        prim for prim in stage.Traverse()
        if prim.GetAttribute("node:type").Get() == "omni.graph.scriptnode.ScriptNode"
    ]
    if len(script_nodes) != 1 or str(script_nodes[0].GetPath()) != SCRIPT_PATH:
        raise ValueError("Scene contains an unexpected set of executable ScriptNodes")
    prim = script_nodes[0]
    if prim.GetAttribute("inputs:usePath").Get() or prim.GetAttribute("inputs:scriptPath").Get():
        raise ValueError("External ScriptNode source paths are not supported")
    script = prim.GetAttribute("inputs:script").Get()
    if not isinstance(script, str) or hashlib.sha256(script.encode()).hexdigest() != SCRIPT_SHA256:
        raise ValueError("Embedded ScriptNode differs from the reviewed source")
    return script


def reviewed_controller_script(original):
    """Build only the reviewed payload; reject changes to either local or remote code."""
    if hashlib.sha256(original.encode()).hexdigest() != SCRIPT_SHA256:
        raise ValueError("Original ScriptNode source hash mismatch")
    controller = Path(__file__).with_name("osc_scene_controller.py").read_text()
    if hashlib.sha256(controller.encode()).hexdigest() != OSC_SHA256:
        raise ValueError("Local OSC source changed; review and update its integrity pin")
    original_index = "if jacobians.shape[0] == body_index:"
    if original.count(original_index) != 1:
        raise ValueError("Unexpected Jacobian compatibility patch location")
    combined = original.replace(
        original_index, "if jacobians.shape[0] == int(physics_view.max_links) - 1:",
    ) + "\n" + controller
    if hashlib.sha256(combined.encode()).hexdigest() != COMBINED_SCRIPT_SHA256:
        raise ValueError("Final executable ScriptNode source hash mismatch")
    return combined
