#!/usr/bin/env python3
"""Inspect a WRDS CRSP Daily Stock File download in data/."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
EXTRACT_DIR = DATA_DIR / "extracted"

CRSP_FIELDS = [
    "permno",
    "permco",
    "ticker",
    "hdrcusip",
    "dlycaldt",
    "dlyprc",
    "dlyret",
    "dlyvol",
    "dlyopen",
    "dlyhigh",
    "dlylow",
    "shrout",
    "cfacpr",
    "cfacshr",
]

QLIB_FIELD_MAPPING = {
    "dlycaldt": "date",
    "dlyopen": "open",
    "dlyhigh": "high",
    "dlylow": "low",
    "dlyprc": "close",
    "dlyvol": "volume",
    "permno": "instrument",
    "ticker": "symbol",
    "dlyret": "return",
    "shrout": "shares_outstanding",
    "cfacpr": "factor_price",
    "cfacshr": "factor_shares",
}


def find_zip_file(data_dir: Path) -> Path:
    zip_files = sorted(data_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(f"未在 {data_dir} 中找到 zip 文件")
    if len(zip_files) > 1:
        print(f"发现 {len(zip_files)} 个 zip 文件，使用最新的: {zip_files[0].name}")
    return zip_files[0]


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"{zip_path.name} 中未找到 csv 文件")

        if len(csv_names) > 1:
            print(f"zip 中包含 {len(csv_names)} 个 csv 文件，将全部解压")

        if marker.exists() and marker.read_text(encoding="utf-8").strip() == zip_path.name:
            print(f"已解压，跳过: {extract_dir}")
        else:
            print(f"正在解压 {zip_path.name} -> {extract_dir}")
            zf.extractall(extract_dir)
            marker.write_text(zip_path.name, encoding="utf-8")

    csv_files = sorted(extract_dir.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"解压后未在 {extract_dir} 中找到 csv 文件")
    if len(csv_files) > 1:
        print(f"发现 {len(csv_files)} 个 csv 文件，使用最大的: {csv_files[0].name}")
    return csv_files[0]


def build_column_lookup(columns: list[str]) -> dict[str, str]:
    return {col.lower(): col for col in columns}


def count_csv_rows(csv_path: Path) -> int | None:
    try:
        with csv_path.open("rb") as handle:
            row_count = sum(1 for _ in handle)
        return max(row_count - 1, 0)
    except OSError as exc:
        print(f"无法统计总行数: {exc}")
        return None


def scan_date_range(csv_path: Path, date_column: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None

    for chunk in pd.read_csv(
        csv_path,
        usecols=[date_column],
        parse_dates=[date_column],
        chunksize=500_000,
        low_memory=False,
    ):
        chunk_min = chunk[date_column].min()
        chunk_max = chunk[date_column].max()
        if pd.notna(chunk_min):
            min_date = chunk_min if min_date is None else min(min_date, chunk_min)
        if pd.notna(chunk_max):
            max_date = chunk_max if max_date is None else max(max_date, chunk_max)

    return min_date, max_date


def print_section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    if not DATA_DIR.exists():
        print(f"data 目录不存在: {DATA_DIR}")
        return 1

    zip_path = find_zip_file(DATA_DIR)
    print(f"找到 zip 文件: {zip_path}")

    extract_target = EXTRACT_DIR / zip_path.stem
    csv_path = extract_zip(zip_path, extract_target)
    print(f"使用 csv 文件: {csv_path}")

    preview = pd.read_csv(csv_path, nrows=1000, low_memory=False)
    column_lookup = build_column_lookup(list(preview.columns))

    print_section("列名")
    for index, column in enumerate(preview.columns, start=1):
        print(f"{index:>3}. {column}")

    print_section("前 5 行数据")
    print(preview.head(5).to_string(index=False))

    print_section("数据概览")
    total_rows = count_csv_rows(csv_path)
    if total_rows is not None:
        print(f"数据总行数: {total_rows:,}")
    else:
        print("数据总行数: 无法获取")

    date_field = column_lookup.get("dlycaldt")
    if date_field:
        print(f"正在扫描日期字段 {date_field} 的范围...")
        min_date, max_date = scan_date_range(csv_path, date_field)
        if min_date is not None and max_date is not None:
            print(f"最早日期: {min_date.date()}")
            print(f"最晚日期: {max_date.date()}")
        else:
            print("日期范围: 无法解析")
    else:
        preview_date_candidates = [col for col in preview.columns if "date" in col.lower() or "dt" in col.lower()]
        if preview_date_candidates:
            print(f"未找到 dlycaldt，预览中的日期相关列: {', '.join(preview_date_candidates)}")
        else:
            print("未找到 dlycaldt 字段，无法输出日期范围")

    found_fields: list[str] = []
    missing_fields: list[str] = []
    for field in CRSP_FIELDS:
        actual = column_lookup.get(field)
        if actual:
            found_fields.append(f"{field} ({actual})")
        else:
            missing_fields.append(field)

    print_section("CRSP 字段检查")
    print("找到的字段:")
    if found_fields:
        for item in found_fields:
            print(f"  + {item}")
    else:
        print("  (无)")

    print()
    print("缺失的字段:")
    if missing_fields:
        for item in missing_fields:
            print(f"  - {item}")
    else:
        print("  (无)")

    print_section("推荐 Qlib 字段映射")
    for crsp_field, qlib_field in QLIB_FIELD_MAPPING.items():
        actual = column_lookup.get(crsp_field)
        if actual:
            print(f"{crsp_field} ({actual}) -> {qlib_field}")
        else:
            print(f"{crsp_field} -> {qlib_field}  [缺失，需补充或从其他数据源合并]")

    extra_close = column_lookup.get("dlyclose")
    if extra_close and "dlyprc" in {field.split()[0] for field in found_fields}:
        print()
        print(
            f"提示: 文件同时包含 {extra_close}。"
            "CRSP 中 DlyPrc 常用于收益计算，DlyClose 为收盘价；"
            "若 Qlib 需要真实收盘价，可考虑 dlyclose -> close。"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
