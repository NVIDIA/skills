"""Owns the ovrtx renderer, render loop stepping, and live attribute writes.

All methods must be called from the render loop thread only.
Refer to the omniverse-realtime-viewer skill → references/ovrtx-rendering/
for the full renderer construction and frame-extraction contract.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_ovrtx = None


def _ov():
    global _ovrtx
    if _ovrtx is None:
        import ovrtx as _m
        _ovrtx = _m
    return _ovrtx


class RendererRuntime:
    """Wraps ovrtx.Renderer. All public methods are render-loop-only."""

    def __init__(self, width: int, height: int, gpu_index: int = 0) -> None:
        self.width = width
        self.height = height
        self.gpu_index = gpu_index
        self._renderer = None
        self._render_product_path: Optional[str] = None
        self._camera_path: Optional[str] = None
        self._stage_loaded = False
        self._frame_index = 0
        self._bgra_buf: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        ov = _ov()
        cfg = ov.RendererConfig(
            sync_mode=True,
            selection_outline_enabled=True,
            cuda_gpu_index=self.gpu_index,
        )
        self._renderer = ov.Renderer(config=cfg)
        log.info("ovrtx.Renderer created (GPU %d)", self.gpu_index)

    def shutdown(self) -> None:
        if self._renderer is not None:
            del self._renderer
            self._renderer = None

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def open_usd_from_string(self, usda: str, render_product_path: str, camera_path: str) -> None:
        assert self._renderer is not None
        self._renderer.open_usd_from_string(usda)
        self._render_product_path = render_product_path
        self._camera_path = camera_path
        self._stage_loaded = True
        self._frame_index = 0
        log.info("Stage loaded  rp=%s  cam=%s", render_product_path, camera_path)

    def reset_stage(self) -> None:
        assert self._renderer is not None
        self._renderer.reset_stage()
        self._stage_loaded = False
        self._render_product_path = None
        self._camera_path = None

    # ------------------------------------------------------------------
    # Frame stepping
    # ------------------------------------------------------------------

    def step_and_get_bgra(self) -> Optional[np.ndarray]:
        """Step one frame; return persistent BGRA8 (H×W×4) buffer or None."""
        if not self._stage_loaded or self._render_product_path is None:
            return None

        frame = self._renderer.step(self._render_product_path)
        if frame is None:
            return None

        rgba = self._extract_ldr_color(frame)
        if rgba is None:
            return None

        # RGBA8 → BGRA8 using persistent buffer to avoid allocations per frame.
        if self._bgra_buf is None or self._bgra_buf.shape != rgba.shape:
            self._bgra_buf = np.empty_like(rgba)
        self._bgra_buf[:, :, 0] = rgba[:, :, 2]  # B ← R
        self._bgra_buf[:, :, 1] = rgba[:, :, 1]  # G ← G
        self._bgra_buf[:, :, 2] = rgba[:, :, 0]  # R ← B
        self._bgra_buf[:, :, 3] = rgba[:, :, 3]  # A ← A

        self._frame_index += 1
        return self._bgra_buf

    def _extract_ldr_color(self, frame) -> Optional[np.ndarray]:
        """Return (H, W, 4) uint8 RGBA from the frame dict."""
        try:
            import torch
            ldr = frame["LdrColor"]
            # DLPack tensor from ovrtx; channel-last (H × W × 4).
            rgba = torch.utils.dlpack.from_dlpack(ldr).cpu().numpy()
            return rgba.astype(np.uint8)
        except Exception:
            pass
        try:
            raw = frame["LdrColor"]
            if hasattr(raw, "__array__"):
                return np.asarray(raw, dtype=np.uint8)
        except Exception:
            pass
        log.warning("Could not extract LdrColor from frame")
        return None

    def warmup(self, frames: int = 8) -> None:
        """Step several frames to complete shader compilation."""
        log.info("Warming up renderer (%d frames)...", frames)
        for _ in range(frames):
            self.step_and_get_bgra()
        log.info("Warmup done (frame_index=%d)", self._frame_index)

    # ------------------------------------------------------------------
    # Live attribute writes — render-loop-only
    # ------------------------------------------------------------------

    def write_camera_transform(self, matrix: np.ndarray) -> None:
        """Write 4×4 world-space camera transform (row-major float64)."""
        if self._renderer is None or self._camera_path is None:
            return
        ov = _ov()
        self._renderer.write_attribute(
            self._camera_path,
            "omni:xform",
            matrix.flatten().tolist(),
            semantic=ov.AttributeSemantic.TRANSFORM_WORLD,
            create_new_prim=False,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stage_loaded(self) -> bool:
        return self._stage_loaded

    @property
    def camera_path(self) -> Optional[str]:
        return self._camera_path

    @property
    def native_renderer(self):
        return self._renderer
