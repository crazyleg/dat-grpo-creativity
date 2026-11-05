#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements-spark.in
uv pip install --python .venv/bin/python --no-deps -e .
