# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged PyTorch Client API conversion template: paired training + evaluation.

Copy and adapt this into a generated ``client.py``. Adapt ``evaluate`` to the
source project's metric and validation loader; keep the source metric name and
averaging denominator rather than inventing new metric semantics. The received
global model is evaluated first so the server can do model selection, and the
metric is returned through ``FLModel.metrics``.

``evaluate`` is a pure function so a generated conversion can be validated
against a toy model and loader without a running FLARE server. The main loop
initializes FLARE, then builds model and training state once before FLARE
rounds; each round only loads received weights into that persistent state.
"""

import math

import torch

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import MetaKey


def evaluate(model, val_loader, device="cpu"):
    """Evaluate ``model`` over ``val_loader`` and return the source-backed metric.

    Replace the accuracy computation with the source project's metric while
    keeping its averaging denominator. Fails closed on empty evaluation data
    instead of reporting a metric from zero samples. Restores the model's prior
    train/eval mode so a later train_one_round() is not left with dropout and
    batchnorm disabled.
    """
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                predictions = model(features).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.numel()
    finally:
        model.train(was_training)
    if total == 0:
        raise RuntimeError("evaluation data is empty; cannot report metrics")
    return correct / total


def _validate_round_num_steps(value):
    """Return a valid positive FedAvg weight from completed local optimizer steps."""
    if isinstance(value, bool):
        raise RuntimeError("train_one_round() must return a positive number of completed optimizer steps")
    try:
        num_steps = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("train_one_round() must return a positive number of completed optimizer steps") from exc
    if not math.isfinite(num_steps) or num_steps <= 0 or not num_steps.is_integer():
        raise RuntimeError("train_one_round() must return a positive number of completed optimizer steps")
    return num_steps


def main(model_factory, train_setup_factory, train_one_round, val_loader, device="cpu", metric_name="accuracy"):
    """Client API round loop with global-model evaluation before local training.

    ``model_factory`` constructs the model with the same constructor args the
    recipe uses. ``train_setup_factory`` constructs stateful training objects
    such as the optimizer, loss, scheduler, and training data loader once for
    the persistent model, after Client API context is available.
    ``train_one_round`` runs the source training loop using that prebuilt state
    and returns the positive number of completed optimizer steps. This value is
    sent as ``NUM_STEPS_CURRENT_ROUND`` so FedAvg weights each site by its
    actual per-round training work.
    """
    # Build all persistent objects before the FLARE round loop. Each round
    # should only load received weights into this state, evaluate, train, and
    # send the updated state dict.
    model = model_factory()
    model.to(device)
    flare.init()
    train_state = train_setup_factory(model, device)

    while flare.is_running():
        input_model = flare.receive()
        model.load_state_dict(input_model.params)

        # Evaluate the received global model first for server-side model selection.
        global_metric = evaluate(model, val_loader, device)

        if flare.is_evaluate():
            flare.send(flare.FLModel(metrics={metric_name: global_metric}))
            continue

        round_num_steps = _validate_round_num_steps(train_one_round(model, train_state))

        params = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        flare.send(
            flare.FLModel(
                params=params,
                metrics={metric_name: global_metric},
                meta={MetaKey.NUM_STEPS_CURRENT_ROUND: round_num_steps},
            )
        )
