#!/usr/bin/env python3
"""Stage 2: first-day normalization + PIT sp500.txt (no dump_bin)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from convert_wrds_stage1 import (
    CHUNK_SIZE,
    DATA_DIR,
    EXTRACT_DIR,
    OUTPUT_COLUMNS,
    OUTPUT_CSV_DIR,
    PROJECT_ROOT,
    STAGING_DIR,
    extract_csv,
    find_zip_file,
    permno_to_filename,
    verify_membership_columns,
)

STAGE1_CSV_DIR = STAGING_DIR / "csv"
NORMALIZED_CSV_DIR = STAGING_DIR / "csv_normalized"
INSTRUMENTS_DIR = STAGING_DIR / "instruments"
SP500_PATH = INSTRUMENTS_DIR / "sp500.txt"
REPORT_PATH = STAGING_DIR / "stage2_validation_report.json"

MEMBERSHIP_READ_COLUMNS = ["PERMNO", "MbrStartDt", "MbrEndDt", "DlyCalDt", "DlyClose", "DlyPrc"]
EXPECTED_SINGLE_SPELL_PERMNO = 871
EXPECTED_MULTI_SPELL_PERMNO = 20


@dataclass
class NormalizationStats:
    files_processed: int = 0
    first_close_not_one: list[str] = field(default_factory=list)
    max_close_factor_vs_stage1_close_error: float = 0.0
    max_close_factor_vs_stage1_ratio_error: float = 0.0
    instruments_with_ratio_error_over_1e9: list[str] = field(default_factory=list)


@dataclass
class MembershipStats:
    total_spells: int = 0
    permnos_single_spell: int = 0
    permnos_multi_spell: int = 0
    permnos_total: int = 0
    matches_expected_audit: bool = False
    multi_spell_examples: list[dict] = field(default_factory=list)


@dataclass
class Stage2Report:
    membership_columns_present: list[str] = field(default_factory=list)
    stage1_input_dir: str = ""
    normalized_output_dir: str = ""
    sp500_path: str = ""
    normalization: dict = field(default_factory=dict)
    membership: dict = field(default_factory=dict)
    sample_instruments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "membership_columns_present": self.membership_columns_present,
            "stage1_input_dir": self.stage1_input_dir,
            "normalized_output_dir": self.normalized_output_dir,
            "sp500_path": self.sp500_path,
            "normalization": self.normalization,
            "membership": self.membership,
            "sample_instruments": self.sample_instruments,
        }


def permno_to_symbol(permno: int) -> str:
    return f"P{int(permno)}"


def resolve_source_csv() -> Path:
    zip_path = find_zip_file(DATA_DIR)
    return extract_csv(zip_path, EXTRACT_DIR / zip_path.stem)


def normalize_instrument(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    first_close = out["close"].iloc[0]
    if pd.isna(first_close) or first_close <= 0:
        raise ValueError("first valid close must be positive")

    for col in ("open", "high", "low", "close", "factor"):
        out[col] = out[col] / first_close
    out["volume"] = out["volume"] * first_close
    return out


def normalize_all_stage1_csvs(norm_stats: NormalizationStats) -> list[Path]:
    if not STAGE1_CSV_DIR.exists():
        raise FileNotFoundError(f"Stage 1 输出目录不存在: {STAGE1_CSV_DIR}")

    if NORMALIZED_CSV_DIR.exists():
        for path in NORMALIZED_CSV_DIR.glob("*.csv"):
            path.unlink()
    NORMALIZED_CSV_DIR.mkdir(parents=True, exist_ok=True)

    output_files: list[Path] = []
    stage1_files = sorted(STAGE1_CSV_DIR.glob("p*.csv"))

    for stage1_path in stage1_files:
        stage1_df = pd.read_csv(stage1_path)
        normalized_df = normalize_instrument(stage1_df)
        out_path = NORMALIZED_CSV_DIR / stage1_path.name
        normalized_df[OUTPUT_COLUMNS].to_csv(out_path, index=False)
        output_files.append(out_path)

        first_close = float(normalized_df["close"].iloc[0])
        if not np.isclose(first_close, 1.0, rtol=0.0, atol=1e-9):
            norm_stats.first_close_not_one.append(stage1_path.stem.upper())

        stage1_ratio = stage1_df["close"] / stage1_df["factor"]
        norm_ratio = normalized_df["close"] / normalized_df["factor"]
        ratio_err = (norm_ratio.values - stage1_ratio.values)
        max_ratio_err = float(np.nanmax(np.abs(ratio_err))) if len(ratio_err) else 0.0
        norm_stats.max_close_factor_vs_stage1_ratio_error = max(
            norm_stats.max_close_factor_vs_stage1_ratio_error,
            max_ratio_err,
        )

        # User-requested check: close_norm/factor_norm vs stage1 close column
        cf_vs_s1 = (normalized_df["close"] / normalized_df["factor"]).values - stage1_df["close"].values
        max_cf_err = float(np.nanmax(np.abs(cf_vs_s1))) if len(cf_vs_s1) else 0.0
        norm_stats.max_close_factor_vs_stage1_close_error = max(
            norm_stats.max_close_factor_vs_stage1_close_error,
            max_cf_err,
        )
        if max_ratio_err > 1e-9:
            norm_stats.instruments_with_ratio_error_over_1e9.append(stage1_path.stem.upper())

        norm_stats.files_processed += 1

    return output_files


def is_valid_trading_row(row: pd.Series) -> bool:
    close = row["DlyClose"]
    if pd.isna(close) or close <= 0:
        return False
    if pd.notna(row.get("DlyPrc")) and row["DlyPrc"] == 0:
        return False
    return True


def build_membership_spells(source_csv: Path) -> tuple[pd.DataFrame, MembershipStats]:
    spell_dates: dict[tuple[int, pd.Timestamp, pd.Timestamp], dict[str, pd.Timestamp]] = defaultdict(
        lambda: {"min_trade": None, "max_trade": None}
    )

    for chunk in pd.read_csv(
        source_csv,
        usecols=MEMBERSHIP_READ_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"], errors="coerce")
        chunk["MbrStartDt"] = pd.to_datetime(chunk["MbrStartDt"], errors="coerce")
        chunk["MbrEndDt"] = pd.to_datetime(chunk["MbrEndDt"], errors="coerce")

        valid = chunk[
            chunk["PERMNO"].notna()
            & chunk["DlyCalDt"].notna()
            & chunk["MbrStartDt"].notna()
            & chunk["MbrEndDt"].notna()
        ].copy()
        valid = valid[valid.apply(is_valid_trading_row, axis=1)]

        for row in valid.itertuples(index=False):
            permno = int(row.PERMNO)
            key = (permno, row.MbrStartDt.normalize(), row.MbrEndDt.normalize())
            trade_date = row.DlyCalDt.normalize()
            bucket = spell_dates[key]
            if bucket["min_trade"] is None or trade_date < bucket["min_trade"]:
                bucket["min_trade"] = trade_date
            if bucket["max_trade"] is None or trade_date > bucket["max_trade"]:
                bucket["max_trade"] = trade_date

    rows: list[dict] = []
    for (permno, mbr_start, mbr_end), bounds in spell_dates.items():
        if bounds["min_trade"] is None or bounds["max_trade"] is None:
            continue
        start = max(mbr_start, bounds["min_trade"])
        end = min(mbr_end, bounds["max_trade"])
        if start > end:
            continue
        rows.append(
            {
                "symbol": permno_to_symbol(permno),
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "permno": permno,
            }
        )

    spells_df = pd.DataFrame(rows).sort_values(["symbol", "start_date"]).reset_index(drop=True)

    spell_counts = spells_df.groupby("symbol").size()
    stats = MembershipStats(
        total_spells=int(len(spells_df)),
        permnos_single_spell=int((spell_counts == 1).sum()),
        permnos_multi_spell=int((spell_counts > 1).sum()),
        permnos_total=int(spell_counts.shape[0]),
        matches_expected_audit=(
            int((spell_counts == 1).sum()) == EXPECTED_SINGLE_SPELL_PERMNO
            and int((spell_counts > 1).sum()) == EXPECTED_MULTI_SPELL_PERMNO
        ),
    )

    multi = spell_counts[spell_counts > 1].head(5)
    for symbol, count in multi.items():
        subset = spells_df[spells_df["symbol"] == symbol][["start_date", "end_date"]]
        stats.multi_spell_examples.append(
            {
                "symbol": symbol,
                "spell_count": int(count),
                "spells": subset.to_dict(orient="records"),
            }
        )

    return spells_df, stats


def write_sp500_txt(spells_df: pd.DataFrame) -> None:
    INSTRUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    output = spells_df[["symbol", "start_date", "end_date"]]
    output.to_csv(SP500_PATH, sep="\t", header=False, index=False)


def build_sample_report(output_files: list[Path], sample_size: int = 5) -> list[dict]:
    rng = np.random.default_rng(42)
    picks = rng.choice(output_files, size=min(sample_size, len(output_files)), replace=False)
    samples: list[dict] = []

    for path in picks:
        stage1_df = pd.read_csv(STAGE1_CSV_DIR / path.name)
        norm_df = pd.read_csv(path)
        samples.append(
            {
                "instrument": path.stem.upper(),
                "file": str(path.relative_to(PROJECT_ROOT)),
                "first_close": float(norm_df["close"].iloc[0]),
                "date_range": {
                    "start": str(norm_df["date"].min()),
                    "end": str(norm_df["date"].max()),
                },
                "preview_first_3_rows": norm_df.head(3)[OUTPUT_COLUMNS].to_dict(orient="records"),
                "close_over_factor_vs_stage1_close_max_abs_error": float(
                    np.max(np.abs(norm_df["close"] / norm_df["factor"] - stage1_df["close"]))
                ),
                "close_over_factor_vs_stage1_ratio_max_abs_error": float(
                    np.max(np.abs(norm_df["close"] / norm_df["factor"] - stage1_df["close"] / stage1_df["factor"]))
                ),
            }
        )

    return samples


def run_stage2() -> Stage2Report:
    source_csv = resolve_source_csv()
    verify_membership_columns(source_csv)

    norm_stats = NormalizationStats()
    print(f"Membership columns verified in: {source_csv}")
    print("Applying first-day normalization to Stage 1 CSVs...")
    output_files = normalize_all_stage1_csvs(norm_stats)

    print("Building point-in-time sp500.txt from MbrStartDt/MbrEndDt...")
    spells_df, membership_stats = build_membership_spells(source_csv)
    write_sp500_txt(spells_df)

    report = Stage2Report(
        membership_columns_present=["MbrStartDt", "MbrEndDt"],
        stage1_input_dir=str(STAGE1_CSV_DIR.relative_to(PROJECT_ROOT)),
        normalized_output_dir=str(NORMALIZED_CSV_DIR.relative_to(PROJECT_ROOT)),
        sp500_path=str(SP500_PATH.relative_to(PROJECT_ROOT)),
        normalization={
            "files_processed": norm_stats.files_processed,
            "all_first_close_equal_one": len(norm_stats.first_close_not_one) == 0,
            "instruments_first_close_not_one": norm_stats.first_close_not_one,
            "max_abs_error_close_over_factor_vs_stage1_close": norm_stats.max_close_factor_vs_stage1_close_error,
            "max_abs_error_close_over_factor_vs_stage1_close_over_factor": norm_stats.max_close_factor_vs_stage1_ratio_error,
            "ratio_invariant_passed": norm_stats.max_close_factor_vs_stage1_ratio_error < 1e-9,
            "note": (
                "Qlib/Yahoo normalization scales the close column; therefore "
                "close_norm/factor_norm equals stage1 close/factor (raw price recovery), "
                "not the stage1 close column itself."
            ),
        },
        membership={
            "total_spells": membership_stats.total_spells,
            "permnos_single_spell": membership_stats.permnos_single_spell,
            "permnos_multi_spell": membership_stats.permnos_multi_spell,
            "permnos_total": membership_stats.permnos_total,
            "expected_audit_single": EXPECTED_SINGLE_SPELL_PERMNO,
            "expected_audit_multi": EXPECTED_MULTI_SPELL_PERMNO,
            "matches_expected_audit": membership_stats.matches_expected_audit,
            "multi_spell_examples": membership_stats.multi_spell_examples,
        },
        sample_instruments=build_sample_report(output_files),
    )

    REPORT_PATH.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_report(report: Stage2Report) -> None:
    print()
    print("=" * 60)
    print("Stage 2 Validation Report")
    print("=" * 60)
    print(f"Membership columns: {', '.join(report.membership_columns_present)}")
    print(f"Stage 1 input: {report.stage1_input_dir}")
    print(f"Normalized output: {report.normalized_output_dir}")
    print(f"sp500.txt: {report.sp500_path}")
    print()
    print("Normalization:")
    for key, value in report.normalization.items():
        print(f"  {key}: {value}")

    print()
    print("Membership spells:")
    for key, value in report.membership.items():
        if key != "multi_spell_examples":
            print(f"  {key}: {value}")

    if report.membership["multi_spell_examples"]:
        print("  multi_spell_examples:")
        for example in report.membership["multi_spell_examples"]:
            print(f"    {example}")

    print()
    print("Sample instruments:")
    for sample in report.sample_instruments:
        print(f"  [{sample['instrument']}] first_close={sample['first_close']}")
        print(f"    date range: {sample['date_range']['start']} -> {sample['date_range']['end']}")
        print(
            "    max |close/factor - stage1 close|: "
            f"{sample['close_over_factor_vs_stage1_close_max_abs_error']:.6e}"
        )
        print(
            "    max |close/factor - stage1 close/factor|: "
            f"{sample['close_over_factor_vs_stage1_ratio_max_abs_error']:.6e}"
        )

    print()
    print(f"Full report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


def main() -> int:
    try:
        report = run_stage2()
        print_report(report)
        return 0
    except Exception as exc:
        print(f"Stage 2 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
