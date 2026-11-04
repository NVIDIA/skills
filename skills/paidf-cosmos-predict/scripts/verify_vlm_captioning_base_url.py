#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the VLM captioning base URL passes an OpenAI /models probe."""

from __future__ import annotations

import argparse

from paidf_common import models_probe_url, verify_captioning_base_url


def main() -> None:
    """Parse CLI args and verify the captioning base URL with a /models probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-captioning-endpoint", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()

    verify_captioning_base_url(args.vlm_captioning_endpoint, args.timeout_seconds)
    print(
        "VLM captioning base URL passed /models preflight probe: "
        f"{models_probe_url(args.vlm_captioning_endpoint)}"
    )


if __name__ == "__main__":
    main()
