#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single Kumo-Forecast inference-HPO trial (basic or DARR mode).

Called by AutoMLRunner as ``python inference_hpo_trial.py --run-config {config_path}``
with ``execution.cwd`` set to the ``Kumo-TS/Kumo-Forecast`` directory and run inside that
project's venv (VirtualEnvSDK).

Mode is chosen by the spec: a non-empty ``dataset.context_csv`` enables DARR, otherwise the
model's direct forecast is scored. The last ``hpo.forecast_horizon`` rows of ``dataset.csv`` are
held out as ground truth. Writes ``{"val_mse": ..., "val_mae": ...}`` to
``<train.output_dir>/metrics.json``.

The AutoML run-config is nested (``dataset:`` / ``hpo:`` / ``train:``) but the SDK's
``load_forecasting_config`` only accepts flat keys or an ``inference:`` section, so the ``hpo``
values are translated into an ``inference:`` mapping first. Empty strings and nulls are dropped
so that ``ckpt`` / ``standardizer_pkl`` keep the SDK defaults.
"""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# execution.cwd is the Kumo-Forecast project directory, so ``sdk`` is importable from it.
sys.path.insert(0, os.getcwd())

from sdk.forecasting import load_forecasting_config, perform_forecasting  # noqa: E402


def _read_frame(path: str, ts_col: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=[ts_col]).set_index(ts_col)


def _build_config(hpo: dict, ts_col: str, target_cols: list[str] | None):
    inference = {k: v for k, v in hpo.items() if v not in ("", None)}
    inference["timestamp_column"] = ts_col
    if target_cols:
        inference["target_column"] = target_cols[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "inference.yaml"
        path.write_text(yaml.safe_dump({"inference": inference}))
        return replace(load_forecasting_config(path), return_all_channels=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.run_config).read_text())
    ds, hpo = cfg["dataset"], cfg["hpo"]
    out = Path(cfg.get("train", {}).get("output_dir", "artifacts/inference_hpo"))
    out.mkdir(parents=True, exist_ok=True)

    ts_col = ds.get("timestamp_col", "timestamp")
    target_cols = [c.strip() for c in ds["target_cols"].split(",")] if ds.get("target_cols") else None

    df = _read_frame(ds["csv"], ts_col)
    horizon = hpo["forecast_horizon"]
    if horizon <= 0 or horizon >= len(df):
        raise ValueError(f"hpo.forecast_horizon must be in [1, {len(df) - 1}], got {horizon}")
    df_input, df_actual = df.iloc[:-horizon], df.iloc[-horizon:]

    forecast_config = _build_config(hpo, ts_col, target_cols)
    kwargs = {}
    if ds.get("context_csv"):  # DARR mode
        kwargs["context_df"] = _read_frame(ds["context_csv"], ts_col).reset_index()
    # perform_forecasting expects the timestamp as a column, not as the index.
    result = perform_forecasting(df=df_input.reset_index(), config=forecast_config, **kwargs)

    # Align predictions to actuals by column name, never by position.
    if target_cols:
        pred_cols = [f"{c}_forecast" for c in target_cols]
        actual_cols = target_cols
    else:
        pred_cols = [c for c in result.columns if c.endswith("_forecast")]
        actual_cols = [c[: -len("_forecast")] for c in pred_cols]
    missing = [c for c in pred_cols if c not in result.columns]
    if missing:
        raise ValueError(f"prediction columns not found: {missing}")
    preds = result[pred_cols].values[:horizon]
    actual = df_actual[actual_cols].values[:horizon]

    val_mse = float(np.mean((preds - actual) ** 2))
    val_mae = float(np.mean(np.abs(preds - actual)))
    (out / "metrics.json").write_text(json.dumps({"val_mse": val_mse, "val_mae": val_mae}))
    mode = "DARR" if ds.get("context_csv") else "basic"
    print(f"mode={mode}  MSE={val_mse:.6f}  MAE={val_mae:.6f}  hpo={ {k: v for k, v in hpo.items() if k not in ('ckpt', 'standardizer_pkl')} }")


if __name__ == "__main__":
    main()
