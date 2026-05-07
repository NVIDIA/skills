<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 --->

<!---

--->
# NVIDIA Docker User Guide for AI Agents
1. Ensure we know the hostname of allocated GPU node, e.g. <node-name>;
2. Get UUID of allocated GPU(s) on GPU node:
   ```bash
   ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null <node-name>
   nvidia-smi -L
   # Output: GPU 0: NVIDIA B200 (UUID: GPU-d8ea7ef9-442e-488f-bd23-d6912699e32d)
   ```
3. Build Docker container on allocated node. E.g., for Transformer integration in TileGym project:
   ```bash
   # Build with source (for development)
   ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null <node-name>
   cd /path/to/project
   docker build --target source -f modeling/transformers/Dockerfile -t model-test:latest .
   ```
4. Run tests in container with specific GPU. CRITICAL: Use GPU UUID (not --gpus all) to avoid multi-tenant conflicts!
   ```bash
   ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null <node-name>
   # Get the UUID from nvidia-smi -L output (step 2)
   docker run --rm --gpus "device=GPU-d8ea7ef9-442e-488f-bd23-d6912699e32d" \
     -v /path/to/project:/workspace/tilegym \
     model-test:latest \
     pytest model_name/test_modeling_model_tilegym.py -v -s
   # Never use:
   # --gpus all (multi-tenant conflict)
   # --gpus 0   (device index, not UUID)
   ```
