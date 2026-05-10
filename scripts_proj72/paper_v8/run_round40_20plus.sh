#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="${PY:-/home/xj/miniconda3/envs/gnn_timing1/bin/python}"
exec "$PY" scripts_proj72/paper_v8/run_arch_opt_round40_20plus.py
