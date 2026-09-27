#!/usr/bin/env python3
"""Wrapper to run holdout-extended Phase 5A dataset build."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY = "/opt/anaconda3/envs/qlib/bin/python"


def main() -> None:
    script = PROJECT_ROOT / "fundamental_pipeline/14_build_fundamental_model_dataset_holdout.py"
    subprocess.run([PY, str(script)], check=True, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
