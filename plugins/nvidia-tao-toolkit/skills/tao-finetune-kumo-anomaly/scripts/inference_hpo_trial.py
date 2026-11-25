#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single Kumo-Anomaly inference-HPO trial.

Called by AutoMLRunner as ``python inference_hpo_trial.py --run-config {config_path}``
with ``execution.cwd`` set to the ``Kumo-TS/Kumo-Anomaly`` directory and run inside that
project's venv (VirtualEnvSDK). Reads the nested run-config, runs
``perform_anomaly_analysis_with_diffusion`` and writes ``{"f1_score": ...}`` to
``<train.output_dir>/metrics.json``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import f1_score

# execution.cwd is the Kumo-Anomaly project directory, so ``sdk`` is importable from it.
sys.path.insert(0, os.getcwd())

from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.run_config).read_text())
    ds, inf = cfg["dataset"], cfg["hpo"]
    out = Path(cfg.get("train", {}).get("output_dir", "artifacts/inference_hpo"))
    out.mkdir(parents=True, exist_ok=True)

    label_col = ds.get("label_col") or "label"
    df = pd.read_csv(ds["csv"]).head(ds.get("eval_rows", 2000))
    if label_col not in df.columns:
        raise ValueError(f"label column {label_col!r} not found in {ds['csv']}; inference HPO needs ground truth")
    labels = df[label_col].values
    # The SDK raises on non-numeric columns: drop the label and keep numeric features only.
    features = df.drop(columns=[label_col]).select_dtypes(include="number")

    results = perform_anomaly_analysis_with_diffusion(
        df=features,
        nsample=inf["nsample"],
        threshold_strategy=inf["threshold_strategy"],
        # "" (the spec default) means: use the pretrained Hugging Face weights / config.
        model_path=inf.get("model_path") or None,
        model_config_path=inf.get("config_path") or None,
    )
    f1 = float(f1_score(labels, results["Anomaly"].values, zero_division=0))
    (out / "metrics.json").write_text(json.dumps({"f1_score": f1}))
    print(f"nsample={inf['nsample']}  strategy={inf['threshold_strategy']}  F1={f1:.4f}")


if __name__ == "__main__":
    main()
