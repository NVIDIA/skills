#!/bin/bash
set -euo pipefail

tests_dir="${HARBOR_TESTS_DIR:-/tests}"
python3 "${tests_dir}/grader.py"
