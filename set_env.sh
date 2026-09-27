#!/usr/bin/env bash
# Source from repo root:  source ./set_env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/01_data:${PROJECT_ROOT}/03_fundamentals_news:${PYTHONPATH:-}"
if [[ -z "${RDAGENT_ROOT:-}" ]]; then
  echo "WARN: set RDAGENT_ROOT to your patched RD-Agent checkout" >&2
fi
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHONPATH includes repo root, 01_data, and 03_fundamentals_news"
