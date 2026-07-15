# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import importlib
import logging
import os
import sys

import numpy as np
import torch
import torchvision.transforms.v2 as T

sys.path.append(os.getcwd())

# Initialize logger with proper configuration
try:
    import nvds_action_detector.ds_logger as ds_logger

    logger = ds_logger.get_logger(__name__)
except ImportError:
    print("######## loading ds_logger failed, Using standard Python logger")
    logger = logging.getLogger(__name__)


def load_ddm_net_model(checkpoint: str, frames_per_side: int = 5, gpu_id: int = 0) -> torch.nn.Module:

    logger.info("Loading DDM-Net model...")
    logger.info(f"Checkpoint path: {checkpoint}")
    logger.info(f"Frames per side: {frames_per_side}")
    logger.info(f"GPU ID: {gpu_id}")

    # raise exception if DDM_BASE_PATH is not set
    logger.info(f"pwd: {os.getcwd()}")
    ddm_base_path = os.getenv("DDM_BASE_PATH", f"{os.getcwd()}/3rdparty/DDM/")
    modeling_package_parent_path = os.path.join(ddm_base_path, "DDM-Net")
    modeling_package_parent_path = os.path.abspath(modeling_package_parent_path)
    logger.info(f"######## modeling_package_parent_path: {modeling_package_parent_path}")

    try:
        sys.path.insert(0, modeling_package_parent_path)
        from modeling.resnetGEBD import resnetGEBD
    finally:
        sys.path.pop(0)

    model = resnetGEBD(
        backbone="resnet50",
        pretrained=False,
        num_classes=2,
        frames_per_side=frames_per_side,
    )

    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        model_prefix = "model."
        module_prefix = "module."
        # clean key names
        for key, value in state_dict.items():
            if key.startswith(model_prefix):
                new_key = key[len(model_prefix) :]
            elif key.startswith(module_prefix):
                new_key = key[len(module_prefix) :]
            else:
                new_key = key
            new_state_dict[new_key] = value

    else:
        logger.info("state_dict doesn't exist. Attempting to load as PyTorch checkpoint...")
        new_state_dict = checkpoint

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        logger.warning("Missing keys in DDM checkppint: %s", missing_keys)
    if unexpected_keys:
        logger.warning("Unexpected keys in DDM checkpoint: %s", unexpected_keys)

    logger.info("DDM-Net model loaded.")

    device = f"cuda:{gpu_id}"
    logger.debug(f"Moving model to GPU {device}")
    model = model.to(device)

    model.eval()

    return model


if __name__ == "__main__":
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--frames_per_side", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    model = load_ddm_net_model(checkpoint=args.checkpoint, frames_per_side=args.frames_per_side, gpu_id=args.gpu_id)
    print("model loaded")
