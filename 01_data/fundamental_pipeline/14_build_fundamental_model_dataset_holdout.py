#!/usr/bin/env python3
"""Extend fundamental_model_dataset through 2025-12-31 (Final Holdout upstream prep)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
P14 = PROJECT_ROOT / "fundamental_pipeline/14_build_fundamental_model_dataset.py"

spec = importlib.util.spec_from_file_location("p14_holdout", P14)
p14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p14)

# Patch after load
p14.LABEL_END = "2025-12-31"
p14.HANDLER_END = "2025-12-31"
p14.SPLIT_RANGES["holdout"] = ("2024-01-01", "2025-12-31")

if __name__ == "__main__":
    raise SystemExit(p14.main())
