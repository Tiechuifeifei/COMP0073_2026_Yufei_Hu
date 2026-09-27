#!/usr/bin/env python3
"""Stage 3: dump_bin + Qlib API validation (no RD-Agent)."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
STAGING_DIR = PROJECT_ROOT / "staging"
NORMALIZED_CSV_DIR = STAGING_DIR / "csv_normalized"
QLIB_DATA_DIR = STAGING_DIR / "qlib_data"
STAGE2_SP500_PATH = STAGING_DIR / "instruments" / "sp500.txt"
REPORT_PATH = STAGING_DIR / "stage3_validation_report.json"

RDAGENT_PYTHON = Path("/opt/anaconda3/envs/rdagent4qlib/bin/python")
DUMP_BIN_PATH = Path(os.environ.get("QLIB_ROOT", "")) / "scripts" / "dump_bin.py"
OUTPUT_COLUMNS = ["open", "high", "low", "close", "volume", "factor"]


@dataclass
class QlibEnvironment:
    conda_env: str
    python_executable: str
    qlib_version: str
    qlib_package_path: str
    dump_bin_path: str
    dump_bin_in_package: bool


@dataclass
class Stage3Report:
    qlib_environment: dict = field(default_factory=dict)
    dump_bin: dict = field(default_factory=dict)
    structure_validation: dict = field(default_factory=dict)
    qlib_init: dict = field(default_factory=dict)
    api_samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "qlib_environment": self.qlib_environment,
            "dump_bin": self.dump_bin,
            "structure_validation": self.structure_validation,
            "qlib_init": self.qlib_init,
            "api_samples": self.api_samples,
        }


def verify_qlib_environment() -> QlibEnvironment:
    if not RDAGENT_PYTHON.exists():
        raise FileNotFoundError(f"RD-Agent Qlib 环境未找到: {RDAGENT_PYTHON}")
    if not DUMP_BIN_PATH.exists():
        raise FileNotFoundError(f"dump_bin.py 未找到: {DUMP_BIN_PATH}")

    probe = subprocess.check_output(
        [
            str(RDAGENT_PYTHON),
            "-c",
            "import qlib, os; print(qlib.__version__); print(os.path.dirname(qlib.__file__))",
        ],
        text=True,
    ).strip().splitlines()
    qlib_version, qlib_package_path = probe[0], probe[1]
    dump_bin_in_package = bool(list(Path(qlib_package_path).rglob("dump_bin.py")))

    return QlibEnvironment(
        conda_env="rdagent4qlib",
        python_executable=str(RDAGENT_PYTHON),
        qlib_version=qlib_version,
        qlib_package_path=qlib_package_path,
        dump_bin_path=str(DUMP_BIN_PATH),
        dump_bin_in_package=dump_bin_in_package,
    )


def run_dump_bin(env: QlibEnvironment) -> dict:
    if QLIB_DATA_DIR.exists():
        shutil.rmtree(QLIB_DATA_DIR)

    cmd = [
        env.python_executable,
        env.dump_bin_path,
        "dump_all",
        "--data_path",
        str(NORMALIZED_CSV_DIR),
        "--qlib_dir",
        str(QLIB_DATA_DIR),
        "--include_fields",
        "open,close,high,low,volume,factor",
        "--date_field_name",
        "date",
        "--file_suffix",
        ".csv",
        "--freq",
        "day",
        "--max_workers",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"dump_bin 失败:\n{result.stderr}\n{result.stdout}")

    QLIB_DATA_DIR.joinpath("instruments").mkdir(parents=True, exist_ok=True)
    shutil.copy2(STAGE2_SP500_PATH, QLIB_DATA_DIR / "instruments" / "sp500.txt")

    return {
        "command": cmd,
        "returncode": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-5:],
    }


def validate_structure() -> dict:
    calendar_path = QLIB_DATA_DIR / "calendars" / "day.txt"
    all_txt_path = QLIB_DATA_DIR / "instruments" / "all.txt"
    sp500_path = QLIB_DATA_DIR / "instruments" / "sp500.txt"
    features_dir = QLIB_DATA_DIR / "features"

    if not calendar_path.exists():
        raise FileNotFoundError("缺少 calendars/day.txt")
    if not all_txt_path.exists():
        raise FileNotFoundError("缺少 instruments/all.txt")
    if not sp500_path.exists():
        raise FileNotFoundError("缺少 instruments/sp500.txt")
    if not features_dir.exists():
        raise FileNotFoundError("缺少 features/ 目录")

    calendar = pd.read_csv(calendar_path, header=None, names=["date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    feature_count = len([p for p in features_dir.iterdir() if p.is_dir()])
    all_count = sum(1 for _ in all_txt_path.open("r", encoding="utf-8"))
    sp500_count = sum(1 for _ in sp500_path.open("r", encoding="utf-8"))

    return {
        "calendar_path": str(calendar_path.relative_to(PROJECT_ROOT)),
        "calendar_rows": int(len(calendar)),
        "calendar_start": str(calendar["date"].min().date()),
        "calendar_end": str(calendar["date"].max().date()),
        "features_instrument_count": feature_count,
        "expected_instrument_count": 891,
        "features_match_expected": feature_count == 891,
        "instruments_all_rows": all_count,
        "instruments_sp500_rows": sp500_count,
        "expected_sp500_spells": 911,
        "sp500_match_expected": sp500_count == 911,
    }


def validate_with_qlib_api(env: QlibEnvironment, sample_size: int = 5) -> tuple[dict, list[dict]]:
    code = f"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_US
from qlib.data import D

provider_uri = {str(QLIB_DATA_DIR)!r}
csv_dir = Path({str(NORMALIZED_CSV_DIR)!r})
fields = {OUTPUT_COLUMNS!r}
qlib_fields = ["$" + f for f in fields]

qlib.init(provider_uri=provider_uri, region=REG_US)

csv_files = sorted(csv_dir.glob("p*.csv"))
rng = random.Random(42)
picked = rng.sample(csv_files, k=min({sample_size}, len(csv_files)))

samples = []
for csv_path in picked:
    symbol = csv_path.stem.upper()
    source = pd.read_csv(csv_path)
    source["date"] = pd.to_datetime(source["date"])
    start = source["date"].min().strftime("%Y-%m-%d")
    end = source["date"].max().strftime("%Y-%m-%d")

    loaded = D.features([symbol], qlib_fields, start_time=start, end_time=end, freq="day")
    loaded = loaded.reset_index()
    loaded.columns = ["instrument", "date"] + fields
    loaded["date"] = pd.to_datetime(loaded["date"])

    merged = source.merge(loaded, on="date", suffixes=("_csv", "_qlib"), how="inner")
    field_errors = {{}}
    worst_volume_row = None
    for field in fields:
        err = (merged[f"{{field}}_csv"] - merged[f"{{field}}_qlib"]).abs()
        if field == "volume":
            f32 = merged["volume_csv"].astype("float64").apply(np.float32).astype("float64")
            f32_err = (merged["volume_qlib"] - f32).abs()
            idx = err.idxmax()
            row = merged.loc[idx]
            rel = float(err.loc[idx] / max(abs(float(row["volume_csv"])), 1.0))
            worst_volume_row = {{
                "permno": symbol,
                "date": str(row["date"].date()),
                "volume_csv": float(row["volume_csv"]),
                "volume_qlib": float(row["volume_qlib"]),
                "volume_csv_as_float32": float(np.float32(row["volume_csv"])),
                "abs_error_csv_vs_qlib": float(err.loc[idx]),
                "abs_error_qlib_vs_float32": float(f32_err.loc[idx]),
                "abs_error_csv_vs_float32": float(abs(float(row["volume_csv"]) - np.float32(row["volume_csv"]))),
                "relative_error_csv_vs_qlib": rel,
            }}
            passed = bool(np.nanmax(f32_err.values) == 0.0)
        else:
            passed = bool(np.nanmax(err.values) < 1e-5)
        field_errors[field] = {{
            "max_abs_error": float(np.nanmax(err.values)),
            "mean_abs_error": float(np.nanmean(err.values)),
            "passed": passed,
        }}

    samples.append({{
        "instrument": symbol,
        "csv_file": str(csv_path),
        "rows_compared": int(len(merged)),
        "date_range": {{"start": start, "end": end}},
        "field_errors": field_errors,
        "worst_volume_row": worst_volume_row,
        "all_fields_passed": all(v["passed"] for v in field_errors.values()),
        "preview_first_3_rows": merged.head(3)[["date"] + [f"{{f}}_qlib" for f in fields]].astype(str).to_dict(orient="records"),
    }})

# global worst volume row across all instruments
global_worst = None
for csv_path in csv_files:
    symbol = csv_path.stem.upper()
    source = pd.read_csv(csv_path)
    source["date"] = pd.to_datetime(source["date"])
    start = source["date"].min().strftime("%Y-%m-%d")
    end = source["date"].max().strftime("%Y-%m-%d")
    loaded = D.features([symbol], ["$volume"], start_time=start, end_time=end, freq="day").reset_index()
    loaded.columns = ["instrument", "date", "volume_qlib"]
    loaded["date"] = pd.to_datetime(loaded["date"])
    merged = source.merge(loaded, on="date", how="inner")
    err = (merged["volume"] - merged["volume_qlib"]).abs()
    idx = err.idxmax()
    row = merged.loc[idx]
    info = {{
        "permno": symbol,
        "date": str(row["date"].date()),
        "volume_csv": float(row["volume"]),
        "volume_qlib": float(row["volume_qlib"]),
        "volume_csv_as_float32": float(np.float32(row["volume"])),
        "abs_error_csv_vs_qlib": float(err.loc[idx]),
        "abs_error_qlib_vs_float32": float(abs(float(row["volume_qlib"]) - np.float32(row["volume"]))),
        "abs_error_csv_vs_float32": float(abs(float(row["volume"]) - np.float32(row["volume"]))),
        "relative_error_csv_vs_qlib": float(err.loc[idx] / max(abs(float(row["volume"])), 1.0)),
    }}
    if global_worst is None or info["abs_error_csv_vs_qlib"] > global_worst["abs_error_csv_vs_qlib"]:
        global_worst = info

print(json.dumps({{"initialized": True, "samples": samples, "global_worst_volume_row": global_worst}}, ensure_ascii=False))
"""
    result = subprocess.check_output([env.python_executable, "-c", code], text=True)
    payload = json.loads(result)
    return (
        {
            "initialized": payload["initialized"],
            "provider_uri": str(QLIB_DATA_DIR),
            "volume_validation_rule": (
                "volume passes only if Qlib value exactly matches float32(CSV value); "
                "any CSV-vs-Qlib gap must equal CSV-vs-float32 quantization error."
            ),
            "global_worst_volume_row": payload["global_worst_volume_row"],
        },
        payload["samples"],
    )


def run_stage3(skip_dump: bool = False) -> Stage3Report:
    env = verify_qlib_environment()
    dump_info = {"skipped": True} if skip_dump else run_dump_bin(env)
    structure = validate_structure()
    qlib_init, api_samples = validate_with_qlib_api(env)

    report = Stage3Report(
        qlib_environment={
            "conda_env": env.conda_env,
            "python_executable": env.python_executable,
            "qlib_version": env.qlib_version,
            "qlib_package_path": env.qlib_package_path,
            "dump_bin_path": env.dump_bin_path,
            "dump_bin_in_installed_package": env.dump_bin_in_package,
            "note": (
                "pyqlib is installed in site-packages; dump_bin.py is run from "
                "Set QLIB_ROOT to your qlib checkout; expected scripts/dump_bin.py."
            ),
        },
        dump_bin={
            **dump_info,
            "provider_uri": str(QLIB_DATA_DIR.relative_to(PROJECT_ROOT)),
            "input_csv_dir": str(NORMALIZED_CSV_DIR.relative_to(PROJECT_ROOT)),
        },
        structure_validation=structure,
        qlib_init=qlib_init,
        api_samples=api_samples,
    )

    REPORT_PATH.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_report(report: Stage3Report) -> None:
    print()
    print("=" * 60)
    print("Stage 3 Validation Report")
    print("=" * 60)
    print("Qlib environment:")
    for key, value in report.qlib_environment.items():
        print(f"  {key}: {value}")

    print()
    print("dump_bin:")
    for key, value in report.dump_bin.items():
        if key != "command":
            print(f"  {key}: {value}")

    print()
    print("Structure validation:")
    for key, value in report.structure_validation.items():
        print(f"  {key}: {value}")

    print()
    print("Qlib init:")
    for key, value in report.qlib_init.items():
        print(f"  {key}: {value}")

    print()
    print("API samples:")
    for sample in report.api_samples:
        status = "PASS" if sample["all_fields_passed"] else "FAIL"
        print(f"  [{sample['instrument']}] {status} rows_compared={sample['rows_compared']}")
        print(f"    date range: {sample['date_range']['start']} -> {sample['date_range']['end']}")
        for field, metrics in sample["field_errors"].items():
            print(
                f"    {field}: max_abs_error={metrics['max_abs_error']:.3e} "
                f"({'PASS' if metrics['passed'] else 'FAIL'})"
            )

    print()
    print(f"Full report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


def main() -> int:
    skip_dump = "--skip-dump" in sys.argv
    try:
        report = run_stage3(skip_dump=skip_dump)
        print_report(report)
        all_pass = all(s["all_fields_passed"] for s in report.api_samples)
        structure_ok = (
            report.structure_validation["features_match_expected"]
            and report.structure_validation["sp500_match_expected"]
        )
        return 0 if all_pass and structure_ok else 1
    except Exception as exc:
        print(f"Stage 3 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
