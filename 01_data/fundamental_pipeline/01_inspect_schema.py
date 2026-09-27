#!/usr/bin/env python3
"""
01_inspect_schema.py

Inspect WRDS library metadata only (no financial observations downloaded).

For each target library:
  - list all tables
  - export db.describe_table() column metadata per table
  - build a consolidated table inventory

Outputs under: fundamental_pipeline/reports/schema/
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import wrds

# Libraries requested for fundamental / CCM linkage exploration.
LIBRARIES: list[str] = [
    "comp_pit",
    "comp_snapshot",
    "comp_urq",
    "comp",
    "crsp_a_ccm",
]

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR / "reports" / "schema"
INVENTORY_PATH = REPORT_DIR / "table_inventory.csv"

# Heuristic name patterns for the terminal summary (metadata only; not assumptions
# about which tables we will eventually use).
QUARTERLY_HINTS = re.compile(
    r"(^|_)(q|fq|fundq|urq|quarter|qtr|qtly|pit)(_|$)|quarterly",
    re.IGNORECASE,
)
LINK_HINTS = re.compile(
    r"(^|_)(link|ccm|map|mapping|bridge|xref|cross)(_|$)|linktable|link_table",
    re.IGNORECASE,
)


def load_local_env(env_path: Path) -> None:
    """Load KEY=VALUE pairs from fundamental_pipeline/.env without printing secrets."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def connect_wrds() -> wrds.Connection:
    """Open a WRDS connection using .env / environment credentials or .pgpass."""
    load_local_env(SCRIPT_DIR / ".env")

    username = os.environ.get("WRDS_USERNAME", "").strip()
    password = os.environ.get("WRDS_PASSWORD", os.environ.get("PGPASSWORD", "")).strip()

    kwargs: dict[str, str] = {}
    if username:
        kwargs["wrds_username"] = username
    if password:
        kwargs["wrds_password"] = password

    print("Connecting to WRDS ...")
    try:
        db = wrds.Connection(**kwargs)
    except EOFError as exc:
        raise RuntimeError(
            "WRDS connection requires credentials in non-interactive mode. "
            "Set WRDS_USERNAME and WRDS_PASSWORD in fundamental_pipeline/.env "
            "or configure ~/.pgpass, then rerun."
        ) from exc
    print("Connected.\n")
    return db


def ensure_report_dir() -> None:
    """Create output directory if missing."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def list_library_tables(db: wrds.Connection, library: str) -> list[str]:
    """Return sorted table names for a WRDS library."""
    tables = db.list_tables(library=library)
    # wrds may return a list or a single string depending on version / library size.
    if isinstance(tables, str):
        tables = [tables]
    return sorted(set(tables))


def save_table_list(library: str, tables: list[str]) -> Path:
    """Write one table name per line."""
    out_path = REPORT_DIR / f"{library}_tables.txt"
    out_path.write_text("\n".join(tables) + ("\n" if tables else ""), encoding="utf-8")
    return out_path


def describe_table_safe(
    db: wrds.Connection, library: str, table: str
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Fetch column metadata via db.describe_table().

    Returns (dataframe, error_message). Never downloads observation rows.
    """
    try:
        desc = db.describe_table(library=library, table=table)
        if desc is None:
            return None, "describe_table returned None"
        df = pd.DataFrame(desc)
        return df, None
    except Exception as exc:  # noqa: BLE001 - keep going for other tables
        return None, str(exc)


def save_table_description(library: str, table: str, desc_df: pd.DataFrame) -> Path:
    """Save describe_table output to CSV."""
    # Use double underscore between library and table in the filename.
    safe_table = table.replace("/", "_")
    out_path = REPORT_DIR / f"{library}__{safe_table}.csv"
    desc_df.to_csv(out_path, index=False)
    return out_path


def inspect_library(db: wrds.Connection, library: str) -> tuple[list[dict], list[str]]:
    """
    Inspect one library end-to-end.

    Returns:
      - inventory rows for table_inventory.csv
      - error messages encountered (non-fatal)
    """
    errors: list[str] = []
    inventory_rows: list[dict] = []

    print(f"=== Library: {library} ===")
    try:
        tables = list_library_tables(db, library)
    except Exception as exc:  # noqa: BLE001
        msg = f"[{library}] list_tables failed: {exc}"
        errors.append(msg)
        print(f"  ERROR: {msg}\n")
        return inventory_rows, errors

    list_path = save_table_list(library, tables)
    print(f"  Tables found: {len(tables)}")
    print(f"  Saved list -> {list_path.relative_to(SCRIPT_DIR)}")

    for table in tables:
        desc_df, err = describe_table_safe(db, library, table)
        if err is not None:
            msg = f"[{library}.{table}] describe_table failed: {err}"
            errors.append(msg)
            print(f"  WARN: {msg}")
            inventory_rows.append(
                {
                    "library": library,
                    "table": table,
                    "number_of_columns": None,
                }
            )
            continue

        out_path = save_table_description(library, table, desc_df)
        n_cols = int(len(desc_df))
        inventory_rows.append(
            {
                "library": library,
                "table": table,
                "number_of_columns": n_cols,
            }
        )
        print(f"  - {table}: {n_cols} columns -> {out_path.name}")

    print()
    return inventory_rows, errors


def classify_possible_tables(inventory_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag likely quarterly-fundamental and link/mapping tables by name pattern."""
    names = inventory_df["table"].astype(str)

    quarterly_mask = names.str.contains(QUARTERLY_HINTS)
    link_mask = names.str.contains(LINK_HINTS)

    quarterly_df = inventory_df[quarterly_mask].copy()
    link_df = inventory_df[link_mask].copy()
    return quarterly_df, link_df


def print_summary(
    inventory_df: pd.DataFrame,
    errors: list[str],
    libraries_requested: list[str],
) -> None:
    """Print human-readable inspection summary."""
    libraries_inspected = sorted(inventory_df["library"].unique().tolist())
    total_tables = len(inventory_df)
    quarterly_df, link_df = classify_possible_tables(inventory_df)

    print("=" * 72)
    print("WRDS SCHEMA INSPECTION SUMMARY")
    print("=" * 72)
    print(f"Libraries requested : {len(libraries_requested)}")
    print(f"Libraries inspected : {len(libraries_inspected)}")
    print(f"Total tables        : {total_tables}")
    print(f"Describe failures   : {int(inventory_df['number_of_columns'].isna().sum())}")
    print()

    print("Libraries:")
    for lib in libraries_requested:
        n = int((inventory_df["library"] == lib).sum())
        status = "ok" if n > 0 or lib in libraries_inspected else "no tables / failed"
        print(f"  - {lib}: {n} tables ({status})")

    print()
    print(f"Possible quarterly / fundamental tables (name heuristic): {len(quarterly_df)}")
    for _, row in quarterly_df.sort_values(["library", "table"]).iterrows():
        print(f"  - {row['library']}.{row['table']} ({row['number_of_columns']} cols)")

    print()
    print(f"Possible link / mapping tables (name heuristic): {len(link_df)}")
    for _, row in link_df.sort_values(["library", "table"]).iterrows():
        print(f"  - {row['library']}.{row['table']} ({row['number_of_columns']} cols)")

    if errors:
        print()
        print(f"Non-fatal errors ({len(errors)}):")
        for msg in errors[:20]:
            print(f"  - {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    print()
    print(f"Inventory CSV: {INVENTORY_PATH.relative_to(SCRIPT_DIR)}")
    print("=" * 72)


def main() -> int:
    ensure_report_dir()

    db = connect_wrds()

    all_rows: list[dict] = []
    all_errors: list[str] = []

    for library in LIBRARIES:
        rows, errs = inspect_library(db, library)
        all_rows.extend(rows)
        all_errors.extend(errs)

    inventory_df = pd.DataFrame(all_rows)
    inventory_df.to_csv(INVENTORY_PATH, index=False)
    print(f"Saved inventory -> {INVENTORY_PATH.relative_to(SCRIPT_DIR)}\n")

    print_summary(inventory_df, all_errors, LIBRARIES)
    return 0 if not inventory_df.empty else 1


if __name__ == "__main__":
    sys.exit(main())
