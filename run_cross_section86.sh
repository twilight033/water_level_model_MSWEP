#!/usr/bin/env bash
# Linux或WSL入口；先预检再运行，保持同一组命令行参数。
set -euo pipefail
cd "$(dirname "$0")"
task_python="python3"
if [[ -v HYDRO_PYTHON ]]; then task_python="$HYDRO_PYTHON"; fi
"$task_python" -u src/cross_section_experiment/runner.py "$@" --preflight
"$task_python" -u src/cross_section_experiment/runner.py "$@"
