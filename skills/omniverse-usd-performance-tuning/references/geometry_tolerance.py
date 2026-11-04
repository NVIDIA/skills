# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical geometry budget for human-scale and factory assets, in millimetres."""
import math

DEFAULT_TOLERANCE_MM = 0.1
TOLERANCE_KEYS = ('tolerance_mm', 'deviation_tolerance_mm', 'max_deviation_mm')


def physical_tolerance_mm(policy=None):
    """Resolve one positive physical budget; metadata and topology are separate.

    Explicit supported aliases remain accepted. Conflicting aliases are rejected
    rather than letting different phases choose different budgets. The budget is
    source-to-delivered geometry, not an allowance to add at every operation.
    """
    policy = {} if policy is None else policy
    values = [policy[key] for key in TOLERANCE_KEYS if key in policy]
    if not values:
        return DEFAULT_TOLERANCE_MM
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('physical tolerance must be positive finite millimetres')
    if len(set(values)) != 1:
        raise ValueError('conflicting physical tolerance aliases')
    return float(values[0])


def local_tolerance_mm(policy, maximum_occurrence_scale=1.0):
    """Bound a shared definition so its largest scaled occurrence meets the budget."""
    scale = maximum_occurrence_scale
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('maximum occurrence scale must be positive and finite')
    return physical_tolerance_mm(policy) / scale


def stage_tolerance(policy, meters_per_unit, maximum_occurrence_scale=1.0):
    """Convert the local millimetre budget to the stage's authored length units."""
    if isinstance(meters_per_unit, bool) or not isinstance(meters_per_unit, (int, float)) or not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError('metersPerUnit must be positive and finite')
    return local_tolerance_mm(policy, maximum_occurrence_scale) / (meters_per_unit * 1000)
