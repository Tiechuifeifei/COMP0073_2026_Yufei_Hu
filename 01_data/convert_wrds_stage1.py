#!/usr/bin/env python3
"""Stage 1: WRDS CRSP CSV -> per-PERMNO Qlib-ready CSV (no .bin, no sp500.txt)."""

from __future__ import annotations

import json
import random
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EXTRACT_DIR = DATA_DIR / "extracted"
STAGING_DIR = PROJECT_ROOT / "staging"
TEMP_PARTS_DIR = STAGING_DIR / "tmp_parts"
OUTPUT_CSV_DIR = STAGING_DIR / "csv"
REPORT_PATH = STAGING_DIR / "stage1_validation_report.json"

READ_COLUMNS = [
    "PERMNO",
    "MbrStartDt",
    "MbrEndDt",
    "DlyCalDt",
    "DlyOpen",
    "DlyHigh",
    "DlyLow",
    "DlyClose",
    "DlyPrc",
    "DlyPrcFlg",
    "DlyVol",
    "DlyCumFacPr",
]

OUTPUT_COLUMNS = ["date", "open", "high", "low", "close", "volume", "factor"]
CHUNK_SIZE = 200_000


@dataclass
class CleanStats:
    input_rows: int = 0
    removed_missing_key: int = 0
    removed_invalid_close: int = 0
    removed_invalid_factor: int = 0
    removed_zero_price: int = 0
    kept_rows: int = 0
    duplicate_dates_resolved: int = 0

    def to_dict(self) -> dict:
        return {
            "input_rows": self.input_rows,
            "removed_missing_key": self.removed_missing_key,
            "removed_invalid_close": self.removed_invalid_close,
            "removed_invalid_factor": self.removed_invalid_factor,
            "removed_zero_price": self.removed_zero_price,
            "removed_total": (
                self.removed_missing_key
                + self.removed_invalid_close
                + self.removed_invalid_factor
                + self.removed_zero_price
            ),
            "kept_rows": self.kept_rows,
            "duplicate_dates_resolved": self.duplicate_dates_resolved,
        }


@dataclass
class Stage1Report:
    source_csv: str
    membership_columns_present: list[str] = field(default_factory=list)
    output_dir: str = ""
    output_csv_count: int = 0
    clean_stats: dict = field(default_factory=dict)
    sample_instruments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_csv": self.source_csv,
            "membership_columns_present": self.membership_columns_present,
            "output_dir": self.output_dir,
            "output_csv_count": self.output_csv_count,
            "clean_stats": self.clean_stats,
            "sample_instruments": self.sample_instruments,
        }


def find_zip_file(data_dir: Path) -> Path:
    zip_files = sorted(data_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(f"未在 {data_dir} 中找到 zip 文件")
    return zip_files[0]


def extract_csv(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"{zip_path.name} 中未找到 csv 文件")

        if marker.exists() and marker.read_text(encoding="utf-8").strip() == zip_path.name:
            print(f"已解压，跳过: {extract_dir}")
        else:
            print(f"正在解压 {zip_path.name} -> {extract_dir}")
            zf.extractall(extract_dir)
            marker.write_text(zip_path.name, encoding="utf-8")

    csv_files = sorted(extract_dir.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"解压后未在 {extract_dir} 中找到 csv 文件")
    return csv_files[0]


def verify_membership_columns(csv_path: Path) -> None:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    required = {"MbrStartDt", "MbrEndDt"}
    missing = required - set(header)
    if missing:
        raise ValueError(
            f"源 CSV 缺少成分股区间字段: {sorted(missing)}。"
            "无法构建 PIT sp500.txt，请先重新下载 WRDS 数据。"
        )


def permno_to_filename(permno: int) -> str:
    return f"p{int(permno)}.csv"


def clean_and_adjust(chunk: pd.DataFrame, stats: CleanStats) -> pd.DataFrame:
    stats.input_rows += len(chunk)

    work = chunk.copy()
    work["DlyCalDt"] = pd.to_datetime(work["DlyCalDt"], errors="coerce")

    missing_key = work["PERMNO"].isna() | work["DlyCalDt"].isna()
    stats.removed_missing_key += int(missing_key.sum())
    work = work.loc[~missing_key].copy()

    invalid_close = work["DlyClose"].isna() | (work["DlyClose"] <= 0)
    stats.removed_invalid_close += int(invalid_close.sum())
    work = work.loc[~invalid_close].copy()

    invalid_factor = work["DlyCumFacPr"].isna() | (work["DlyCumFacPr"] <= 0)
    stats.removed_invalid_factor += int(invalid_factor.sum())
    work = work.loc[~invalid_factor].copy()

    zero_price = work["DlyPrc"].fillna(0) == 0
    stats.removed_zero_price += int(zero_price.sum())
    work = work.loc[~zero_price].copy()

    if work.empty:
        return work

    factor = 1.0 / work["DlyCumFacPr"]
    out = pd.DataFrame(
        {
            "date": work["DlyCalDt"].dt.strftime("%Y-%m-%d"),
            "open": work["DlyOpen"] / work["DlyCumFacPr"],
            "high": work["DlyHigh"] / work["DlyCumFacPr"],
            "low": work["DlyLow"] / work["DlyCumFacPr"],
            "close": work["DlyClose"] / work["DlyCumFacPr"],
            "volume": work["DlyVol"] / factor,
            "factor": factor,
            "_raw_close": work["DlyClose"],
            "_permno": work["PERMNO"].astype(int),
        }
    )

    stats.kept_rows += len(out)
    return out


def append_parts(processed: pd.DataFrame) -> None:
    for permno, group in processed.groupby("_permno", sort=False):
        part_path = TEMP_PARTS_DIR / permno_to_filename(int(permno))
        group.drop(columns=["_permno"]).to_csv(
            part_path,
            mode="a",
            header=not part_path.exists(),
            index=False,
        )


def finalize_outputs(stats: CleanStats) -> list[Path]:
    OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []

    for part_path in sorted(TEMP_PARTS_DIR.glob("p*.csv")):
        df = pd.read_csv(part_path, parse_dates=["date"])
        before = len(df)
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        stats.duplicate_dates_resolved += before - len(df)

        out_path = OUTPUT_CSV_DIR / part_path.name
        df[OUTPUT_COLUMNS].to_csv(out_path, index=False)
        output_files.append(out_path)

    return output_files


def build_sample_report(output_files: list[Path], sample_size: int = 5) -> list[dict]:
    rng = random.Random(42)
    samples = rng.sample(output_files, k=min(sample_size, len(output_files)))
    report_samples: list[dict] = []

    for path in samples:
        df = pd.read_csv(path)
        if df.empty:
            continue

        part_df = pd.read_csv(TEMP_PARTS_DIR / path.name)
        recovered_raw_close = part_df["close"] / part_df["factor"]
        abs_err = (recovered_raw_close - part_df["_raw_close"]).abs()
        max_err = float(abs_err.max()) if len(abs_err) else 0.0
        mean_err = float(abs_err.mean()) if len(abs_err) else 0.0

        preview = df.head(3)[OUTPUT_COLUMNS].to_dict(orient="records")
        report_samples.append(
            {
                "instrument": path.stem.upper(),
                "file": str(path.relative_to(PROJECT_ROOT)),
                "rows": int(len(df)),
                "date_range": {
                    "start": str(df["date"].min()),
                    "end": str(df["date"].max()),
                },
                "raw_close_recovery": {
                    "max_abs_error": max_err,
                    "mean_abs_error": mean_err,
                    "passed": max_err < 1e-6,
                },
                "preview_first_3_rows": preview,
            }
        )

    return report_samples


def run_stage1() -> Stage1Report:
    zip_path = find_zip_file(DATA_DIR)
    csv_path = extract_csv(zip_path, EXTRACT_DIR / zip_path.stem)
    verify_membership_columns(csv_path)

    if TEMP_PARTS_DIR.exists():
        shutil.rmtree(TEMP_PARTS_DIR)
    TEMP_PARTS_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_CSV_DIR.exists():
        shutil.rmtree(OUTPUT_CSV_DIR)

    stats = CleanStats()
    print(f"读取源 CSV（分块 size={CHUNK_SIZE:,}）: {csv_path}")

    for chunk in pd.read_csv(
        csv_path,
        usecols=READ_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        processed = clean_and_adjust(chunk, stats)
        if not processed.empty:
            append_parts(processed)

    output_files = finalize_outputs(stats)
    sample_reports = build_sample_report(output_files)

    report = Stage1Report(
        source_csv=str(csv_path.relative_to(PROJECT_ROOT)),
        membership_columns_present=["MbrStartDt", "MbrEndDt"],
        output_dir=str(OUTPUT_CSV_DIR.relative_to(PROJECT_ROOT)),
        output_csv_count=len(output_files),
        clean_stats=stats.to_dict(),
        sample_instruments=sample_reports,
    )

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_report(report: Stage1Report) -> None:
    print()
    print("=" * 60)
    print("Stage 1 Validation Report")
    print("=" * 60)
    print(f"Source CSV: {report.source_csv}")
    print(f"Membership columns: {', '.join(report.membership_columns_present)}")
    print(f"Output directory: {report.output_dir}")
    print(f"Output CSV files: {report.output_csv_count}")
    print()
    print("Cleaning summary:")
    for key, value in report.clean_stats.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    print()
    print("Random sample instruments:")
    for sample in report.sample_instruments:
        print(f"\n  [{sample['instrument']}] {sample['file']}")
        print(f"    rows: {sample['rows']}")
        print(f"    date range: {sample['date_range']['start']} -> {sample['date_range']['end']}")
        recovery = sample["raw_close_recovery"]
        status = "PASS" if recovery["passed"] else "FAIL"
        print(
            f"    raw_close recovery: {status} "
            f"(max_abs_error={recovery['max_abs_error']:.2e}, mean={recovery['mean_abs_error']:.2e})"
        )
        print("    preview (first 3 rows):")
        for row in sample["preview_first_3_rows"]:
            print(f"      {row}")

    print()
    print(f"Full report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


def main() -> int:
    try:
        report = run_stage1()
        print_report(report)
        return 0
    except Exception as exc:
        print(f"Stage 1 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
