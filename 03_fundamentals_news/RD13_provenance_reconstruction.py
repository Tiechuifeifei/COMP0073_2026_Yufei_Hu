#!/usr/bin/env python3
"""RD13 provenance audit and RD13_v2 reconstruction.

Audit the 13 IC-surpass RD factors, keep v1 alias metadata for reference, and
build RD13_v2_CORRECTED from workspace outputs plus formulas on daily_pv.h5.
Writes corrected artefacts alongside the originals."""

from __future__ import annotations
import os

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "data/fundamental_experiments/RD13_provenance"
V1_ROOT = OUT_ROOT / "RD13_v1_ALIAS_CONTAMINATED"
V2_ROOT = OUT_ROOT / "RD13_v2_CORRECTED"

INVENTORY_V1 = PROJECT_ROOT / "data/fundamental_experiments/R05_rd_fundamental_model_datasets/R05_rd13_feature_inventory.csv"
SOTA_INV = PROJECT_ROOT / "docs/experiment_log/0710_log/phase2_experiment_data/sota_factor_inventory_ic_surpass.csv"
V1_PARQUET_SRC = PROJECT_ROOT / "staging/elite13_factor_pred/combined_elite13_factors.parquet"
DAILY_PV = (Path(os.environ["RDAGENT_ROOT"]) / "git_ignore_folder/factor_implementation_source_data/daily_pv.h5")
MODELLING_END = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")
NEAR_DUP_CORR = 0.9999


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def sha256_series(s: pd.Series) -> str:
    arr = np.ascontiguousarray(s.to_numpy(dtype=np.float64, na_value=np.nan))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_col(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    return re.sub(r"_+", "_", s).strip("_") or "factor"


def flatten_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[1] if c[0] == "feature" else c[0] for c in out.columns]
    else:
        out.columns = [str(c) for c in out.columns]
    return out


def series_stats(s: pd.Series) -> dict[str, Any]:
    x = s.dropna()
    return {
        "dtype": str(s.dtype),
        "non_null": int(s.notna().sum()),
        "null": int(s.isna().sum()),
        "nunique": int(s.nunique(dropna=True)),
        "mean": float(x.mean()) if len(x) else None,
        "std": float(x.std()) if len(x) else None,
        "p01": float(x.quantile(0.01)) if len(x) else None,
        "p25": float(x.quantile(0.25)) if len(x) else None,
        "p50": float(x.quantile(0.50)) if len(x) else None,
        "p75": float(x.quantile(0.75)) if len(x) else None,
        "p99": float(x.quantile(0.99)) if len(x) else None,
        "sha256": sha256_series(s),
    }


def load_daily_pv() -> pd.DataFrame:
    df = pd.read_hdf(DAILY_PV, key="data")
    df = df.sort_index(level=["instrument", "datetime"])
    dates = df.index.get_level_values("datetime")
    assert dates.max() < HOLDOUT_START, f"daily_pv extends into holdout: {dates.max()}"
    return df


def load_workspace_factor(ws_path: Path) -> tuple[pd.Series | None, list[str], str | None]:
    result = ws_path / "result.h5"
    factor_py = ws_path / "factor.py"
    if not result.exists():
        return None, [], None
    hdf = pd.read_hdf(result, key="data")
    cols = [str(c) for c in hdf.columns]
    if len(cols) != 1:
        return None, cols, factor_py.read_text(encoding="utf-8") if factor_py.exists() else None
    s = hdf.iloc[:, 0]
    s.name = cols[0]
    code = factor_py.read_text(encoding="utf-8") if factor_py.exists() else None
    return s, cols, code


def read_factor_py_summary(code: str | None) -> str:
    if not code:
        return "MISSING"
    lines = [ln.strip() for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return " | ".join(lines[:8])


# ---------------------------------------------------------------------------
# Formula implementations (ex-ante: information through date t only)
# ---------------------------------------------------------------------------

def compute_all_v2_factors(pv: pd.DataFrame) -> pd.DataFrame:
    """Compute 13 corrected RD13 features aligned to daily_pv index."""
    g = pv.groupby(level="instrument", sort=False)
    close = pv["$close"]
    high = pv["$high"]
    low = pv["$low"]
    volume = pv["$volume"]

    ret_simple = g["$close"].transform(lambda x: x.pct_change())
    log_ret = g["$close"].transform(lambda x: np.log(x / x.shift(1)))

    out = pd.DataFrame(index=pv.index)

    # Loop 0
    out["rd13_10_day_momentum"] = g["$close"].transform(lambda x: x / x.shift(10) - 1)

    out["rd13_20_day_realized_volatility"] = ret_simple.groupby(level="instrument", sort=False).transform(
        lambda x: x.rolling(window=20, min_periods=20).std(ddof=1)
    )

    out["rd13_5_day_volume_deviation"] = g["$volume"].transform(
        lambda x: x / x.rolling(window=5, min_periods=5).mean() - 1.0
    )

    hl_range = high - low
    out["rd13_10_day_price_range_ratio"] = hl_range / hl_range.groupby(level="instrument", sort=False).transform(
        lambda x: x.rolling(window=10, min_periods=10).mean()
    )

    # Loop 1
    mom_10d = out["rd13_10_day_momentum"]
    vol_20d = log_ret.groupby(level="instrument", sort=False).transform(
        lambda x: x.rolling(window=20, min_periods=20).std(ddof=1)
    )
    out["rd13_volatility_adjusted_10d_momentum"] = mom_10d / vol_20d

    out["rd13_5d_short_term_reversal"] = g["$close"].transform(lambda x: -(x / x.shift(5) - 1.0))

    out["rd13_daily_amihud_illiquidity"] = ret_simple.abs() / volume.replace(0, np.nan)

    # Loop 3 — rolling beta (match workspace logic)
    market_ret = log_ret.groupby(level="datetime").transform("mean")
    panel = pd.DataFrame({"r": log_ret, "market_ret": market_ret}, index=pv.index).reset_index()
    beta_parts = []
    for inst, grp in panel.groupby("instrument", sort=False):
        grp = grp.sort_values("datetime")
        r = grp["r"]
        mr = grp["market_ret"]
        cov = r.rolling(window=20, min_periods=20).cov(mr)
        var = mr.rolling(window=20, min_periods=20).var()
        beta = cov / var
        beta_parts.append(
            pd.DataFrame({"datetime": grp["datetime"], "instrument": grp["instrument"], "beta": beta.values})
        )
    beta_df = pd.concat(beta_parts, ignore_index=True)
    beta_df = beta_df.set_index(["datetime", "instrument"]).sort_index()
    out["rd13_rolling_beta_20d"] = beta_df["beta"].reindex(pv.index)

    eps = ret_simple - out["rd13_rolling_beta_20d"] * market_ret
    out["rd13_idiosyncratic_volatility_10d"] = eps.groupby(level="instrument", sort=False).transform(
        lambda x: x.rolling(window=10, min_periods=10).std(ddof=1)
    )

    mom_5d = g["$close"].transform(lambda x: x / x.shift(5) - 1.0)
    out["rd13_cs_momentum_rank_5d"] = mom_5d.groupby(level="datetime").rank(method="average", pct=True)

    # Loop 7
    win = (ret_simple > 0).astype(float)
    out["rd13_win_rate_5d"] = win.groupby(level="instrument", sort=False).transform(
        lambda x: x.rolling(window=5, min_periods=5).mean()
    )

    up_vol = volume * (ret_simple > 0).astype(float)
    out["rd13_upday_volume_ratio_5d"] = (
        up_vol.groupby(level="instrument", sort=False).transform(lambda x: x.rolling(5, min_periods=5).sum())
        / volume.groupby(level="instrument", sort=False).transform(lambda x: x.rolling(5, min_periods=5).sum())
    )

    prev_close = g["$close"].shift(1)
    tr = pd.concat(
        [
            (high - low).rename("hl"),
            (high - prev_close).abs().rename("hc"),
            (low - prev_close).abs().rename("lc"),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.groupby(level="instrument", sort=False).transform(lambda x: x.rolling(14, min_periods=14).mean())
    mean_close_14 = close.groupby(level="instrument", sort=False).transform(
        lambda x: x.rolling(14, min_periods=14).mean()
    )
    out["rd13_atr_norm_close_14d"] = atr_14 / mean_close_14

    return out.astype(np.float32)


WORKSPACE_CANONICAL = {
    "rd13_10_day_momentum": "9cbbe98fa62341d092b31d9efb66e8e2",
    "rd13_volatility_adjusted_10d_momentum": "9729bffba4e04b1db07ebe896b38c3f3",
    "rd13_rolling_beta_20d": "ea9b4556e7d5458a862e8df441de1e9a",
    "rd13_win_rate_5d": "1a79408064cd47988a4a1eb7f6a33b40",
}


def audit_provenance(inv: pd.DataFrame, sota: pd.DataFrame, v1_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sota_form = dict(zip(sota["factor_name"], sota["formulation"]))
    rows = []
    ws_reuse: dict[str, list[str]] = {}
    value_hashes: dict[str, str] = {}

    for _, r in inv.iterrows():
        out = r["output_feature_name"]
        ws = Path(r["workspace"])
        ws_id = r["workspace_id"]
        ws_reuse.setdefault(ws_id, []).append(r["factor_name"])

        ws_series, hdf_cols, code = load_workspace_factor(ws)
        v1_series = v1_raw[r["parquet_column"]]
        value_hashes[out] = sha256_series(v1_series)

        implemented = "UNKNOWN"
        semantics_match = None
        if ws_series is not None and len(hdf_cols) == 1:
            implemented = f"single HDF column '{hdf_cols[0]}' from result.h5"
            if out in WORKSPACE_CANONICAL and WORKSPACE_CANONICAL[out] == ws_id:
                ws_flat = ws_series.reset_index()
                ws_flat["datetime"] = pd.to_datetime(ws_flat["datetime"]).dt.normalize()
                cmp = v1_raw[["datetime", "instrument", r["parquet_column"]]].merge(
                    ws_flat.rename(columns={hdf_cols[0]: "_ws"}),
                    on=["datetime", "instrument"],
                    how="inner",
                )
                if len(cmp):
                    semantics_match = bool(np.allclose(
                        cmp[r["parquet_column"]].to_numpy(),
                        cmp["_ws"].to_numpy(),
                        rtol=1e-5, atol=1e-6, equal_nan=True,
                    ))
                else:
                    semantics_match = False
            else:
                semantics_match = False
        elif ws_series is None:
            implemented = "result.h5 missing or multi-column"
            semantics_match = False

        st = series_stats(v1_series)
        rows.append({
            "inventory_row": int(r["inventory_row"]),
            "declared_factor_name": r["factor_name"],
            "declared_formula": sota_form.get(r["factor_name"], ""),
            "adoption_loop": int(r["adoption_loop"]),
            "workspace": str(ws),
            "workspace_id": ws_id,
            "factor_py_path": str(ws / "factor.py"),
            "actual_implemented_formula_summary": read_factor_py_summary(code),
            "actual_implementation_note": implemented,
            "result_h5_path": str(ws / "result.h5"),
            "actual_hdf_column_names": hdf_cols,
            "selected_hdf_column": hdf_cols[0] if len(hdf_cols) == 1 else None,
            "assigned_parquet_column_v1": r["parquet_column"],
            "output_feature_name": out,
            "dtype": st["dtype"],
            "non_null_count_v1": st["non_null"],
            "sha256_v1": st["sha256"],
            "semantics_match_implementation": semantics_match,
            "is_workspace_canonical_factor": out in WORKSPACE_CANONICAL and WORKSPACE_CANONICAL[out] == ws_id,
            "workspace_reused_across_inventory_rows": len(ws_reuse[ws_id]) > 1,
        })

    prov = pd.DataFrame(rows)

    # exact duplicate groups on v1
    hash_to_feats: dict[str, list[str]] = {}
    for out, h in value_hashes.items():
        hash_to_feats.setdefault(h, []).append(out)
    exact_groups = [{ "sha256": h, "features": feats} for h, feats in hash_to_feats.items() if len(feats) > 1]

    alias_of: dict[str, str | None] = {}
    for _, row in prov.iterrows():
        out = row["output_feature_name"]
        if row["is_workspace_canonical_factor"]:
            alias_of[out] = None
        else:
            canonical = [f for f in hash_to_feats.get(row["sha256_v1"], []) if f in WORKSPACE_CANONICAL]
            alias_of[out] = canonical[0] if canonical else hash_to_feats[row["sha256_v1"]][0]

    prov["exact_alias_of"] = prov["output_feature_name"].map(alias_of)

    # near duplicates on v2 preview later; placeholder in meta
    meta = {
        "exact_duplicate_groups_v1": exact_groups,
        "workspace_reuse_map": ws_reuse,
        "genuinely_independent_v1_count": len(exact_groups) + len([h for h, f in hash_to_feats.items() if len(f) == 1]),
        "value_hashes_v1": value_hashes,
    }
    return prov, meta


def near_duplicate_groups(panel: pd.DataFrame, features: list[str]) -> list[dict]:
    groups: list[dict] = []
    corr = panel[features].corr(method="pearson")
    seen: set[tuple[str, str]] = set()
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            if (a, b) in seen:
                continue
            seen.add((a, b))
            c = corr.loc[a, b]
            if pd.notna(c) and abs(c) >= NEAR_DUP_CORR:
                groups.append({"feature_a": a, "feature_b": b, "pearson": float(c)})
    return groups


def build_v1_snapshot(v1_mapped: pd.DataFrame, inv: pd.DataFrame) -> None:
    V1_ROOT.mkdir(parents=True, exist_ok=True)
    v1_mapped.to_parquet(V1_ROOT / "rd13_v1_panel.parquet", index=False)
    manifest = {
        "label": "RD13_v1_ALIAS_CONTAMINATED",
        "preserved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_parquet": str(V1_PARQUET_SRC),
        "source_parquet_sha256": sha256_file(V1_PARQUET_SRC),
        "source_inventory": str(INVENTORY_V1),
        "note": "Frozen snapshot of v1 mapped RD13 columns; originals not modified.",
        "features": inv["output_feature_name"].tolist(),
    }
    (V1_ROOT / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")


def main() -> int:
    for d in (V1_ROOT, V2_ROOT):
        d.mkdir(parents=True, exist_ok=True)

    inv = pd.read_csv(INVENTORY_V1)
    sota = pd.read_csv(SOTA_INV)
    v1_raw = flatten_parquet(pd.read_parquet(V1_PARQUET_SRC))
    rename = dict(zip(inv["parquet_column"], inv["output_feature_name"]))
    v1_mapped = v1_raw[["datetime", "instrument"] + list(rename.keys())].rename(columns=rename)

    # A. Provenance audit
    provenance, prov_meta = audit_provenance(inv, sota, v1_raw)
    provenance.to_csv(OUT_ROOT / "RD13_provenance_table.csv", index=False)
    (OUT_ROOT / "RD13_provenance_meta.json").write_text(
        json.dumps(json_safe(prov_meta), indent=2), encoding="utf-8",
    )

    build_v1_snapshot(v1_mapped, inv)

    # B. Reconstruct v2
    pv = load_daily_pv()
    v2_computed = compute_all_v2_factors(pv)
    features = inv["output_feature_name"].tolist()
    v2_panel = v2_computed[features].reset_index()
    v2_panel["datetime"] = pd.to_datetime(v2_panel["datetime"]).dt.normalize()
    assert v2_panel["datetime"].max() <= MODELLING_END

    # Prefer genuine workspace outputs for canonical 4 (should match computed closely)
    recon_notes: list[dict] = []
    for feat, ws_id in WORKSPACE_CANONICAL.items():
        ws_path = Path(inv.loc[inv["workspace_id"] == ws_id, "workspace"].iloc[0])
        ws_series, _, _ = load_workspace_factor(ws_path)
        computed = v2_computed[feat]
        if ws_series is not None:
            aligned = ws_series.reindex(computed.index)
            max_diff = float((aligned - computed).abs().max(skipna=True))
            recon_notes.append({
                "feature": feat, "source": "workspace_result_h5_verified",
                "max_abs_diff_vs_formula": max_diff,
            })
            v2_computed[feat] = aligned.astype(np.float32)
        else:
            recon_notes.append({"feature": feat, "source": "formula_fallback", "max_abs_diff_vs_formula": None})

    invalid_feats = [f for f in features if f not in WORKSPACE_CANONICAL]
    for feat in invalid_feats:
        recon_notes.append({
            "feature": feat,
            "source": "formula_implementation_daily_pv",
            "method": "declared_formula_direct",
        })

    v2_panel = v2_computed[features].reset_index()
    v2_panel["datetime"] = pd.to_datetime(v2_panel["datetime"]).dt.normalize()

    # C. Versioned outputs
    v2_parquet = V2_ROOT / "rd13_v2_panel.parquet"
    v2_panel.to_parquet(v2_parquet, index=False)
    v2_h5 = V2_ROOT / "rd13_reference_v2.h5"
    v2_computed.to_hdf(v2_h5, key="data")

    # corrected inventory
    v2_inv_rows = []
    mapping_rows = []
    col_hashes = {}
    for _, r in inv.iterrows():
        out = r["output_feature_name"]
        st = series_stats(v2_panel[out])
        col_hashes[out] = st["sha256"]
        v2_inv_rows.append({
            **r.to_dict(),
            "parquet_column_v2": out,
            "reconstruction_source": next(
                (n["source"] for n in recon_notes if n["feature"] == out), "unknown"
            ),
            "sha256_v2": st["sha256"],
            "non_null_v2": st["non_null"],
        })
        mapping_rows.append({
            "output_feature_name": out,
            "v1_parquet_column": r["parquet_column"],
            "v2_parquet_column": out,
            "v1_sha256": provenance.loc[provenance["output_feature_name"] == out, "sha256_v1"].iloc[0],
            "v2_sha256": st["sha256"],
            "v1_v2_identical": provenance.loc[provenance["output_feature_name"] == out, "sha256_v1"].iloc[0] == st["sha256"],
            "reconstruction_action": (
                "preserved_canonical" if out in WORKSPACE_CANONICAL else "formula_reconstructed"
            ),
        })

    v2_inv = pd.DataFrame(v2_inv_rows)
    v2_inv.to_csv(V2_ROOT / "RD13_v2_inventory.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(V2_ROOT / "RD13_v1_to_v2_mapping.csv", index=False)
    (V2_ROOT / "RD13_v2_column_hashes.json").write_text(
        json.dumps(json_safe(col_hashes), indent=2), encoding="utf-8",
    )
    (V2_ROOT / "RD13_v2_reconstruction_notes.json").write_text(
        json.dumps(json_safe(recon_notes), indent=2), encoding="utf-8",
    )

    manifest_v2 = {
        "label": "RD13_v2_CORRECTED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "daily_pv_source": str(DAILY_PV),
        "daily_pv_sha256": sha256_file(DAILY_PV),
        "modelling_end": str(MODELLING_END.date()),
        "holdout_2024_accessed": False,
        "rows": int(len(v2_panel)),
        "features": features,
        "genuinely_independent_v1": prov_meta["genuinely_independent_v1_count"],
        "genuinely_independent_v2": len({h for h in col_hashes.values()}),
        "artifacts": {
            "panel_parquet": str(v2_parquet.relative_to(PROJECT_ROOT)),
            "reference_h5": str(v2_h5.relative_to(PROJECT_ROOT)),
            "inventory": "RD13_v2_inventory.csv",
            "mapping": "RD13_v1_to_v2_mapping.csv",
        },
        "file_hashes": {
            "rd13_v2_panel.parquet": sha256_file(v2_parquet),
            "rd13_reference_v2.h5": sha256_file(v2_h5),
        },
    }
    (V2_ROOT / "RD13_v2_manifest.json").write_text(json.dumps(json_safe(manifest_v2), indent=2), encoding="utf-8")

    # D. Validation
    v2_exact_dups = [{ "sha256": h, "features": [f for f, hh in col_hashes.items() if hh == h]}
                     for h in set(col_hashes.values()) if sum(1 for hh in col_hashes.values() if hh == h) > 1]
    v2_near = near_duplicate_groups(v2_panel, features)
    compare_rows = []
    for f in features:
        m = v1_mapped[["datetime", "instrument", f]].merge(
            v2_panel[["datetime", "instrument", f]], on=["datetime", "instrument"], suffixes=("_v1", "_v2"),
        )
        both = m[f + "_v1"].notna() & m[f + "_v2"].notna()
        compare_rows.append({
            "feature": f,
            "pearson_v1_v2": float(m.loc[both, f + "_v1"].corr(m.loc[both, f + "_v2"])) if both.any() else None,
            "row_equal_v1_v2": int((m.loc[both, f + "_v1"] == m.loc[both, f + "_v2"]).sum()) if both.any() else 0,
            "sha_equal": bool(m[f + "_v1"].equals(m[f + "_v2"])),
        })

    miss = v2_panel[features].isna().mean().rename("missing_rate").reset_index(drop=True)
    miss_report = pd.DataFrame({"feature": features, "missing_rate": [v2_panel[f].isna().mean() for f in features]})
    corr_v2 = v2_panel[features].corr()
    corr_v2.to_csv(V2_ROOT / "RD13_v2_correlation_matrix.csv")
    miss_report.to_csv(V2_ROOT / "RD13_v2_missingness.csv", index=False)
    pd.DataFrame(compare_rows).to_csv(V2_ROOT / "RD13_v1_v2_comparison.csv", index=False)

    coverage = {
        "date_min": str(v2_panel["datetime"].min().date()),
        "date_max": str(v2_panel["datetime"].max().date()),
        "n_rows": int(len(v2_panel)),
        "n_instruments": int(v2_panel["instrument"].nunique()),
        "holdout_rows": int((v2_panel["datetime"] >= HOLDOUT_START).sum()),
    }

    unrecoverable = [f for f in invalid_feats if v2_panel[f].notna().sum() == 0]
    validation = {
        "all_formulas_implemented": len(unrecoverable) == 0,
        "exact_duplicate_groups_v2": v2_exact_dups,
        "near_duplicate_pairs_v2": v2_near,
        "no_suffix_alias_different_semantics": len(v2_exact_dups) == 0,
        "ex_ante_daily_pv_max_date": str(pv.index.get_level_values("datetime").max()),
        "coverage": coverage,
        "unrecoverable_features": unrecoverable,
        "compare_v1": compare_rows,
    }
    (V2_ROOT / "RD13_v2_validation.json").write_text(json.dumps(json_safe(validation), indent=2), encoding="utf-8")

    # Report
    report_lines = [
        "# RD13 Provenance and Reconstruction Audit",
        "",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## A. Provenance summary",
        "",
        f"- Declared factors: **13**",
        f"- Genuinely independent in v1: **{prov_meta['genuinely_independent_v1_count']}**",
        f"- Exact duplicate groups in v1: **{len(prov_meta['exact_duplicate_groups_v1'])}**",
        f"- Genuinely independent in v2: **{manifest_v2['genuinely_independent_v2']}**",
        "",
        "## B. v1 exact duplicate groups",
        "",
        "```json",
        json.dumps(json_safe(prov_meta["exact_duplicate_groups_v1"]), indent=2),
        "```",
        "",
        "## C. v2 near-duplicate pairs (|r| >= 0.9999)",
        "",
        "```json",
        json.dumps(json_safe(v2_near[:20]), indent=2),
        "```",
        "",
        "## D. Validation",
        "",
        "```json",
        json.dumps(json_safe({k: validation[k] for k in ['all_formulas_implemented', 'exact_duplicate_groups_v2', 'coverage', 'unrecoverable_features']}), indent=2),
        "```",
        "",
    ]
    (OUT_ROOT / "RD13_provenance_reconstruction_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    # Gate
    if unrecoverable:
        gate = "RD13_RECONSTRUCTION_BLOCKED"
    elif manifest_v2["genuinely_independent_v2"] == 13 and len(v2_exact_dups) == 0:
        gate = "RD13_V2_READY"
    else:
        gate = "RD13_PARTIALLY_RECONSTRUCTED"

    (OUT_ROOT / "RD13_gate.txt").write_text(gate + "\n", encoding="utf-8")
    print(gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
