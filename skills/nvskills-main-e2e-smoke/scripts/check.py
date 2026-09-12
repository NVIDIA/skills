#!/usr/bin/env python3
"""Return the deterministic result used by the main workflow smoke test."""

from __future__ import annotations


SUCCESS_MESSAGE = "NVSKILLS_MAIN_E2E_OK"


def main() -> int:
    print(SUCCESS_MESSAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
