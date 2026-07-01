#! /bin/bash
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

################################################################################
# Reference start_server.sh for NVDS SOP Inference Microservice
# Source: https://github.com/NVIDIA/sop-monitoring-blueprints/blob/main/microservices/sop-inference-bp/start_server.sh
# Local:  sop-inference-bp/start_server.sh
#
# This is the Docker ENTRYPOINT script. It launches the FastAPI server
# via the nvds_action_detector.api_server module.
################################################################################

set -e

python3 -m nvds_action_detector.api_server
