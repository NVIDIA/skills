# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged custom-aggregation template: a step-weighted ``ModelAggregator``.

Copy and adapt this into a generated ``aggregators.py`` when the conversion
needs custom aggregation. Wire it through the recipe ``aggregator=`` parameter
in ``job.py`` with the matching ``aggregator_data_kind`` and parameter transfer
settings. This uses the product extension point rather than a skill-owned
algorithm table, and it fits the standard ``FLModel`` exchange contract by
carrying both params and finite numeric or boolean metrics into the aggregated
``FLModel``. If any client omits ``FLModel.metrics`` entirely, the round returns
no metrics; when every client provides a metrics dictionary, each metric key is
averaged over only the clients that reported that key. It needs no client-side
change beyond sending step-count metadata.
"""

import math

from nvflare.apis.dxo import MetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


def _finite_number(value, allow_bool=False, allow_string=True):
    if isinstance(value, bool):
        return float(value) if allow_bool else None
    if value is None or (isinstance(value, str) and not allow_string):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _step_weight(model: FLModel) -> float:
    # Mirror the product's base_fedavg._get_num_steps_weight: use the client's
    # step count only when it is a finite positive number; otherwise fall back
    # to 1.0. Reject bool (a bool is an int in Python) and non-numeric/None, and
    # never let negative or non-finite metadata corrupt the weighted average.
    weight = _finite_number((model.meta or {}).get(MetaKey.NUM_STEPS_CURRENT_ROUND))
    if weight is None:
        return 1.0
    return weight if weight > 0 else 1.0


def _merge_log_weight(old_log_weight, weight):
    new_log_weight = math.log(weight)
    if old_log_weight == new_log_weight:
        return old_log_weight + math.log(2.0), 0.5
    max_log_weight = max(old_log_weight, new_log_weight)
    total_log_weight = max_log_weight + math.log1p(
        math.exp(min(old_log_weight, new_log_weight) - max_log_weight)
    )
    return total_log_weight, math.exp(new_log_weight - total_log_weight)


def _materialize(value):
    # Disk-offloaded params (enable_tensor_disk_offload=True) may arrive as lazy
    # references rather than in-memory tensors. materialize() loads the tensor
    # from disk before the weighted-sum math, mirroring nvflare's
    # WeightedAggregationHelper.add(); a plain tensor has no materialize() and is
    # returned unchanged.
    materialize_fn = getattr(value, "materialize", None)
    return materialize_fn() if callable(materialize_fn) else value


class WeightedAggregator(ModelAggregator):
    """Average client updates weighted by each client's local step count."""

    def __init__(self):
        super().__init__()
        self.reset_stats()

    def reset_stats(self):
        self._param_mean = {}
        # Per-key weight so a parameter present in only some clients is averaged
        # over just those clients (not diluted by the full round weight), and a
        # key missing from the first client does not raise KeyError.
        self._key_log_weight = {}
        self._metric_mean = {}
        self._metric_log_weight = {}
        self._all_metrics = True
        self._params_type = None
        self._accepted = 0

    def _add_weighted_metric(self, name, value, weight):
        number = _finite_number(value, allow_bool=True, allow_string=False)
        if number is None:
            return
        if name not in self._metric_mean:
            self._metric_mean[name] = number
            self._metric_log_weight[name] = math.log(weight)
            return

        total_log_weight, new_fraction = _merge_log_weight(self._metric_log_weight[name], weight)
        mean = math.fsum(
            (
                self._metric_mean[name] * (1.0 - new_fraction),
                number * new_fraction,
            )
        )
        if not math.isfinite(mean):
            self.log_warning(
                self.fl_ctx,
                f"{self.__class__.__name__} ignored metric {name!r} because its weighted mean is non-finite.",
            )
            return
        self._metric_mean[name] = mean
        self._metric_log_weight[name] = total_log_weight

    def accept_model(self, model: FLModel):
        weight = _step_weight(model)
        self._params_type = model.params_type
        for key, value in model.params.items():
            value = _materialize(value)
            if key in self._param_mean:
                total_log_weight, new_fraction = _merge_log_weight(self._key_log_weight[key], weight)
                self._param_mean[key] = (
                    self._param_mean[key] * (1.0 - new_fraction) + value * new_fraction
                )
                self._key_log_weight[key] = total_log_weight
            else:
                self._param_mean[key] = value * 1.0
                self._key_log_weight[key] = math.log(weight)
        if model.metrics is None:
            self._all_metrics = False
        elif self._all_metrics:
            for name, value in model.metrics.items():
                self._add_weighted_metric(name, value, weight)
        self._accepted += 1
        self.log_info(
            self.fl_ctx,
            f"{self.__class__.__name__} accepted model #{self._accepted} "
            f"(weight={weight}, tensors={len(model.params)}, metrics={len(model.metrics or {})})",
        )

    def aggregate_model(self) -> FLModel:
        if not self._param_mean:
            raise RuntimeError("no client models accepted this round")
        averaged = dict(self._param_mean)
        metrics = dict(self._metric_mean)
        if not self._all_metrics:
            self.log_warning(
                self.fl_ctx,
                f"{self.__class__.__name__} will not return aggregated metrics because at least one "
                "accepted client model omitted FLModel.metrics.",
            )
            metrics = {}
        elif not metrics:
            self.log_warning(
                self.fl_ctx,
                f"{self.__class__.__name__} accepted {self._accepted} models but found no finite numeric "
                "or boolean FLModel.metrics to aggregate.",
            )
        result = FLModel(params=averaged, params_type=self._params_type, metrics=metrics or None)
        self.log_info(
            self.fl_ctx,
            f"{self.__class__.__name__} aggregated {self._accepted} models "
            f"into {len(averaged)} tensors and {len(metrics)} metrics",
        )
        self.reset_stats()
        return result
