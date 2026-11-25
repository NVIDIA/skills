# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch
from model import CheckpointNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    model = CheckpointNetwork()
    # This source call is intentionally unsafe. The conversion must replace it
    # with weight-only loading rather than copying unrestricted pickle loading.
    model.load_state_dict(torch.load(args.checkpoint, weights_only=False))


if __name__ == "__main__":
    main()
