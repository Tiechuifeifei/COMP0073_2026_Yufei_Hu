#!/usr/bin/env bash
# Source from repo root:  source ./set_env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
if [[ -z "${RDAGENT_ROOT:-}" ]]; then
  echo "WARN: set RDAGENT_ROOT to your patched RD-Agent checkout" >&2
fi
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PYTHONPATH includes repo root (fundamental_experiments → 03_fundamentals_news, etc.)"
