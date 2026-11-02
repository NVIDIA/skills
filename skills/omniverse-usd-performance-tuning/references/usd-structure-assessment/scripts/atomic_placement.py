# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common placement proof with exact content and a physical occurrence budget."""
import math
import importlib.util
from pathlib import Path


def common_placement(stage, source, representative, mapping, policy=None):
    """Return common local affine placement within the physical budget, or None.

    Correspondence and geometry/appearance equality are the caller's obligation.
    Recheck all geometry frames, rejecting reset stacks and singular transforms.
    """
    from pxr import UsdGeom, Gf
    import numpy as np
    spec = importlib.util.spec_from_file_location('atomic_geometry_budget', Path(__file__).resolve().parents[2]/'geometry_tolerance.py')
    budget = importlib.util.module_from_spec(spec); spec.loader.exec_module(budget)
    tolerance_mm = budget.physical_tolerance_mm(policy)
    if mapping.get(source) != representative or len(set(mapping.values())) != len(mapping):
        return None
    cache = UsdGeom.XformCache()
    source_frame = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(source))
    representative_frame = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(representative))
    for frame in (source_frame, representative_frame):
        if not all(math.isfinite(v) for row in frame for v in row) or frame.GetDeterminant() == 0:
            return None
    delta = None
    count = 0
    max_linear = 0.0
    max_translation_mm = 0.0
    max_position_mm = 0.0
    for left, right in sorted(mapping.items()):
        prim, other = stage.GetPrimAtPath(left), stage.GetPrimAtPath(right)
        if not prim or not other:
            return None
        for node in (prim, other):
            xf = UsdGeom.Xformable(node)
            if xf and (xf.GetResetXformStack() or any(op.GetAttr().GetTimeSamples() for op in xf.GetOrderedXformOps())):
                return None
        if not prim.IsA(UsdGeom.Gprim):
            continue
        a = cache.GetLocalToWorldTransform(prim) * source_frame.GetInverse()
        b = cache.GetLocalToWorldTransform(other) * representative_frame.GetInverse()
        if (not all(math.isfinite(v) for frame in (a,b) for row in frame for v in row)
                or a.GetDeterminant() == 0 or b.GetDeterminant() == 0):
            return None
        if delta is None:
            delta = Gf.Matrix4d(1) if a == b else b.GetInverse() * a
        reconstructed = b * delta
        linear = max(abs(reconstructed[i][j]-a[i][j]) for i in range(3) for j in range(3))
        translation_mm = max(abs(reconstructed[3][j]-a[3][j]) for j in range(3))*UsdGeom.GetStageMetersPerUnit(stage)*1000
        # Bound the spatial effect of the affine residual in the actual source
        # occurrence. Matrix coefficients themselves do not have length units.
        residual = np.asarray(reconstructed) - np.asarray(a)
        if np.any(residual[:3, :3]):
            if prim.IsA(UsdGeom.Mesh):
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                if points.size == 0 or not np.all(np.isfinite(points)): return None
                low, high = points.min(axis=0), points.max(axis=0)
            elif prim.IsA(UsdGeom.Cube):
                half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
                low, high = np.full(3, -half), np.full(3, half)
            elif prim.IsA(UsdGeom.Cylinder):
                cylinder = UsdGeom.Cylinder(prim)
                high = np.full(3, float(cylinder.GetRadiusAttr().Get()))
                high['XYZ'.index(str(cylinder.GetAxisAttr().Get()))] = float(cylinder.GetHeightAttr().Get()) / 2
                low = -high
            else: return None
            corners = np.array([[low[0] if i else high[0], low[1] if j else high[1], low[2] if k else high[2], 1.]
                                for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        else:
            corners = np.array([[0., 0., 0., 1.]])
        errors = (corners @ residual @ np.asarray(source_frame))[:, :3]
        position_mm = float(np.linalg.norm(errors, axis=1).max()) * UsdGeom.GetStageMetersPerUnit(stage) * 1000
        if not math.isfinite(position_mm) or position_mm > tolerance_mm: return None
        max_position_mm = max(max_position_mm, position_mm)
        max_linear = max(max_linear,linear)
        max_translation_mm = max(max_translation_mm,translation_mm)
        count += 1
    if (delta is None or not all(math.isfinite(v) for row in delta for v in row)
            or any(delta[i][3] != 0 for i in range(3)) or delta[3][3] != 1):
        return None
    return {'method': 'common_affine_at_verifier_precision', 'matrix': [list(row) for row in delta],
            'geometry_count': count, 'max_linear_residual': max_linear,
            'max_translation_residual_mm': max_translation_mm,
            'max_position_error_mm': max_position_mm, 'tolerance_mm': tolerance_mm}
