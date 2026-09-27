"""Build daily_pv-compatible OHLCV from Qlib for holdout extension (2024–2025)."""

from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QLIB_DATA_DIR = PROJECT_ROOT / "staging" / "qlib_data"
SP500_PATH = QLIB_DATA_DIR / "instruments" / "sp500.txt"
DAILY_PV_FROZEN = Path(
    str(Path(os.environ["RDAGENT_ROOT"]) / "git_ignore_folder/factor_implementation_source_data/daily_pv.h5")
)

FIELDS = ["$open", "$high", "$low", "$close", "$volume"]


def _init_qlib() -> None:
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_US, kernels=1)


def load_sp500_instruments() -> list[str]:
    inst: list[str] = []
    for line in SP500_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        inst.append(line.split("\t")[0])
    return inst


def load_frozen_daily_pv() -> pd.DataFrame:
    df = pd.read_hdf(DAILY_PV_FROZEN, key="data")
    df = df.sort_index()
    return df


def build_qlib_pv(start: str, end: str, instruments: list[str] | None = None) -> pd.DataFrame:
    """Return MultiIndex (datetime, instrument) frame with $open/$high/$low/$close/$volume."""
    _init_qlib()
    from qlib.data import D

    if instruments is None:
        instruments = load_sp500_instruments()
    raw = D.features(instruments, FIELDS, start_time=start, end_time=end, freq="day")
    raw = raw.reset_index()
    raw["datetime"] = pd.to_datetime(raw["datetime"]).dt.normalize()
    raw["instrument"] = raw["instrument"].astype(str)
    raw = raw.set_index(["datetime", "instrument"]).sort_index()
    for c in FIELDS:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw


def extend_daily_pv_through(end: str = "2025-12-31") -> pd.DataFrame:
    """Concatenate frozen daily_pv (≤2023-12-29) with qlib extension from 2024-01-01."""
    frozen = load_frozen_daily_pv()
    frozen_max = frozen.index.get_level_values("datetime").max()
    ext_start = (frozen_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if pd.Timestamp(ext_start) > pd.Timestamp(end):
        return frozen
    inst = sorted(frozen.index.get_level_values("instrument").unique())
    ext = build_qlib_pv(ext_start, end, inst)
    # Align columns
    for c in FIELDS:
        if c not in ext.columns:
            raise ValueError(f"missing qlib field {c}")
    combined = pd.concat([frozen, ext], axis=0)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined
