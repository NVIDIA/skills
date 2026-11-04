# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical output names emitted by the pinned TAO 7.2 PAS evaluator."""

PAS_METRICS_FILENAME = "nvidia_pas_metrics.csv"
PAS_METRICS_AGGREGATE_FILENAME = "nvidia_pas_metrics_aggregate.csv"
PAS_METRICS_WEIGHTED_AGGREGATE_FILENAME = (
    "nvidia_pas_metrics_weighted_aggregate.csv"
)

__all__ = [
    "PAS_METRICS_AGGREGATE_FILENAME",
    "PAS_METRICS_FILENAME",
    "PAS_METRICS_WEIGHTED_AGGREGATE_FILENAME",
]
