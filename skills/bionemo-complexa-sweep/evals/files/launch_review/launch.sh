#!/usr/bin/env bash
set -euo pipefail

# Run from the root of a prepared Proteina-Complexa checkout.
SWEEPER_PATH=${1:?Usage: bash launch.sh /absolute/path/to/sweeper.yaml}

for target in 02_PDL1 22_DerF21; do
    python3 script_utils/generate_inference_configs.py \
        --config_name search_binder_local_pipeline \
        --sweeper "$SWEEPER_PATH" \
        --run_name binder_comparison \
        --override "generation.task_name=$target" generation.args.nsteps=400
done

for config in configs/inference_configs/inf_*_binder_comparison.yaml; do
    complexa design "$config"
done
