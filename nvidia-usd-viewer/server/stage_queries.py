"""USD stage queries using ovrtx native prim discovery.

Prefer ovrtx.query_prims() over importing pxr for basic prim discovery.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def get_hierarchy(renderer) -> dict:
    """Return a JSON-serializable hierarchy dict from ovrtx native queries."""
    if renderer is None:
        return {"prims": []}
    try:
        prims = renderer.query_prims()
        return {
            "prims": [
                {"path": p.path, "type": p.type_name}
                for p in (prims or [])
            ]
        }
    except Exception as exc:
        log.warning("query_prims failed: %s", exc)
        return {"prims": []}
