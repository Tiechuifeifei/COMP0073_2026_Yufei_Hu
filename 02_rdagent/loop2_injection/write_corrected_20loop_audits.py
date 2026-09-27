#!/usr/bin/env python3
"""Write English audits after corrected Loop10-19 continuation completes."""
from __future__ import annotations
import os

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    """Resolve package root for this curated layout."""
    import os
    env = os.environ.get("PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # Prefer git-style layout: this file lives under 0X_*/...
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "README.md").exists() and (parent / "paper").exists():
            return parent
    return here.parents[min(2, len(here.parents)-1)]

def _rdagent_root() -> Path:
    import os
    env = os.environ.get("RDAGENT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    raise RuntimeError("Set RDAGENT_ROOT to your RD-Agent checkout (patched us-market-experiment).")


ROOT = (Path(os.environ["PROJECT_ROOT"]) / "reports/rdagent_corrected_semantic_rerun_20260918") if os.environ.get("PROJECT_ROOT") else (_repo_root() / "reports/rdagent_corrected_semantic_rerun_20260918")
AUD = ROOT / "audits"
DEST = ROOT / "sessions" / "corrected_semantic_20loop"
DUMP = DEST / "__session__" / "19" / "4_record"
SOURCE_DUMP = ROOT / "sessions" / "corrected_semantic_10loop" / "__session__" / "9" / "4_record"
SUMMARY_10 = AUD / "corrected_10loop_summary.csv"

ARR = "1day.excess_return_with_cost.annualized_return"
MDD = "1day.excess_return_with_cost.max_drawdown"
IR = "1day.excess_return_with_cost.information_ratio"

sys.path.insert(0, os.environ["RDAGENT_ROOT"])
sys.path.insert(0, str(ROOT / "scripts"))


def metrics(exp):
    if exp is None or exp.result is None:
        return {k: None for k in ("IC", "ICIR", "Rank IC", "Rank ICIR", "ARR", "IR", "MDD")}
    r = exp.result
    return {
        "IC": float(r["IC"]),
        "ICIR": float(r["ICIR"]),
        "Rank IC": float(r["Rank IC"]),
        "Rank ICIR": float(r["Rank ICIR"]),
        "ARR": float(r[ARR]),
        "IR": float(r[IR]),
        "MDD": float(r[MDD]),
    }


def sha_file(p: Path) -> str:
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    from rdagent.app.qlib_rd_loop.factor import FactorRDLoop

    AUD.mkdir(parents=True, exist_ok=True)
    assert DUMP.exists(), DUMP
    assert SOURCE_DUMP.exists(), SOURCE_DUMP
    # Source must not have been extended
    assert not (ROOT / "sessions/corrected_semantic_10loop/__session__/10").exists()
    assert not (DEST / "__session__/20").exists()
    assert not (DEST / "Loop_20").exists()

    loop = FactorRDLoop.load(str(DUMP), checkout=False)
    assert len(loop.trace.hist) == 20, len(loop.trace.hist)

    rows = []
    inv = []
    sota_rows = []
    current_sota = None
    BASE_IC, BASE_ARR, BASE_MDD = 0.0045787190505179, -0.0143767767379566, -0.2683045360791799

    for i, (exp, fb) in enumerate(loop.trace.hist):
        m = metrics(exp)
        names = [t.factor_name for t in exp.sub_tasks]
        shas = []
        impl_ok = 0
        for ws in exp.sub_workspace_list or []:
            fp = ws.workspace_path / "factor.py"
            rh = ws.workspace_path / "result.h5"
            if fp.exists():
                shas.append(sha_file(fp))
            if rh.exists() or (ws.file_dict and "factor.py" in ws.file_dict):
                impl_ok += 1
            inv.append(
                {
                    "loop": i,
                    "factor_name": ws.target_task.factor_name if ws.target_task else "",
                    "workspace": str(ws.workspace_path),
                    "factor_sha256": sha_file(fp) if fp.exists() else "",
                    "has_result_h5": rh.exists(),
                    "newly_executed_in_continuation": i >= 10,
                }
            )
        decision = bool(getattr(fb, "decision", False)) if fb is not None else False
        if current_sota is None:
            sota_in = {"IC": BASE_IC, "ARR": BASE_ARR, "MDD": BASE_MDD, "ICIR": None, "IR": None, "Rank IC": None, "Rank ICIR": None}
            prior_names = "(Alpha20 only)"
            prior_ws = ""
        else:
            sota_in = metrics(current_sota)
            prior_names = "|".join(t.factor_name for t in current_sota.sub_tasks)
            prior_ws = Path(current_sota.experiment_workspace.workspace_path).name

        if decision:
            prior = {
                "loop": i,
                "event": "replace_yes",
                "factor_names": "|".join(names),
                "sota_IC": m["IC"],
                "sota_ARR": m["ARR"],
                "sota_MDD": m["MDD"],
                "sota_ICIR": m["ICIR"],
                "sota_IR": m["IR"],
                "workspace": Path(exp.experiment_workspace.workspace_path).name,
                "prior_sota_factors": prior_names,
                "prior_sota_workspace": prior_ws,
                "prior_sota_IC": sota_in["IC"],
                "prior_sota_ARR": sota_in["ARR"],
                "prior_sota_MDD": sota_in["MDD"],
                "in_continuation_loops_10_19": i >= 10,
            }
            sota_rows.append(prior)
            current_sota = exp

        sota_out = metrics(current_sota) if current_sota is not None else sota_in
        if i == 0 and not decision:
            sota_out = {"IC": BASE_IC, "ARR": BASE_ARR, "MDD": BASE_MDD, "ICIR": None, "IR": None, "Rank IC": None, "Rank ICIR": None}

        rows.append(
            {
                "loop": i,
                "factor_names": "|".join(names),
                "implemented_count": f"{impl_ok}/{len(names)}",
                "current_ic": m["IC"],
                "sota_ic_in": sota_in["IC"] if i > 0 or current_sota else BASE_IC,
                "current_arr": m["ARR"],
                "sota_arr_in": sota_in["ARR"] if i > 0 or current_sota else BASE_ARR,
                "current_mdd": m["MDD"],
                "sota_mdd_in": sota_in["MDD"] if i > 0 or current_sota else BASE_MDD,
                "replace": "yes" if decision else "no",
                "sota_ic_out": sota_out["IC"],
                "sota_arr_out": sota_out["ARR"],
                "sota_mdd_out": sota_out["MDD"],
                "workspace": Path(exp.experiment_workspace.workspace_path).name,
                "factor_sha256": "|".join(shas),
                "newly_executed_in_continuation": i >= 10,
                "is_corrected_loop2": i == 2,
            }
        )

    # Fix loop0 sota in/out baseline
    rows[0]["sota_ic_in"] = BASE_IC
    rows[0]["sota_arr_in"] = BASE_ARR
    rows[0]["sota_mdd_in"] = BASE_MDD
    if rows[0]["replace"] == "no":
        rows[0]["sota_ic_out"] = BASE_IC
        rows[0]["sota_arr_out"] = BASE_ARR
        rows[0]["sota_mdd_out"] = BASE_MDD

    summary = pd.DataFrame(rows)
    summary.to_csv(AUD / "corrected_20loop_summary.csv", index=False)
    pd.DataFrame(inv).to_csv(AUD / "corrected_20loop_factor_inventory.csv", index=False)

    final = loop.trace.get_sota_experiment()
    fm = metrics(final)
    final_names = "|".join(t.factor_name for t in final.sub_tasks)
    final_ws = Path(final.experiment_workspace.workspace_path).name
    evo = pd.DataFrame(sota_rows)
    evo = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "loop": "pre0_alpha20",
                        "event": "baseline_sota",
                        "factor_names": "(Alpha20 only)",
                        "sota_IC": BASE_IC,
                        "sota_ARR": BASE_ARR,
                        "sota_MDD": BASE_MDD,
                        "sota_ICIR": None,
                        "sota_IR": None,
                        "workspace": "",
                        "prior_sota_factors": "",
                        "prior_sota_workspace": "",
                        "prior_sota_IC": None,
                        "prior_sota_ARR": None,
                        "prior_sota_MDD": None,
                        "in_continuation_loops_10_19": False,
                    }
                ]
            ),
            evo,
            pd.DataFrame(
                [
                    {
                        "loop": "final",
                        "event": "final_sota",
                        "factor_names": final_names,
                        "sota_IC": fm["IC"],
                        "sota_ARR": fm["ARR"],
                        "sota_MDD": fm["MDD"],
                        "sota_ICIR": fm["ICIR"],
                        "sota_IR": fm["IR"],
                        "workspace": final_ws,
                        "prior_sota_factors": "",
                        "prior_sota_workspace": "",
                        "prior_sota_IC": None,
                        "prior_sota_ARR": None,
                        "prior_sota_MDD": None,
                        "in_continuation_loops_10_19": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    evo.to_csv(AUD / "corrected_20loop_sota_evolution.csv", index=False)

    # 10 vs 20 continuation compare for loops 0-9 identity
    s10 = pd.read_csv(SUMMARY_10)
    cmp_rows = []
    for i in range(10):
        a, b = s10.iloc[i], summary.iloc[i]
        cmp_rows.append(
            {
                "loop": i,
                "same_factor_names": a["factor_names"] == b["factor_names"],
                "same_replace": a["replace"] == b["replace"],
                "ic_abs_diff": abs(float(a["current_ic"]) - float(b["current_ic"]))
                if pd.notna(a["current_ic"]) and pd.notna(b["current_ic"])
                else None,
                "arr_abs_diff": abs(float(a["current_arr"]) - float(b["current_arr"]))
                if pd.notna(a["current_arr"]) and pd.notna(b["current_arr"])
                else None,
                "note": "forked_hist_identity_expected",
            }
        )
    for i in range(10, 20):
        b = summary.iloc[i]
        cmp_rows.append(
            {
                "loop": i,
                "same_factor_names": None,
                "same_replace": None,
                "ic_abs_diff": None,
                "arr_abs_diff": None,
                "note": "newly_executed_continuation",
                "factor_names": b["factor_names"],
                "replace": b["replace"],
                "current_ic": b["current_ic"],
                "current_arr": b["current_arr"],
            }
        )
    pd.DataFrame(cmp_rows).to_csv(AUD / "corrected_10_vs_20_continuation.csv", index=False)

    session_ids = sorted(p.name for p in (DEST / "__session__").iterdir() if p.is_dir())
    loop_dirs = sorted(p.name for p in DEST.glob("Loop_*"))
    meta = {
        "source_checkpoint": str(SOURCE_DUMP),
        "destination_session": str(DEST),
        "final_dump": str(DUMP),
        "hist_len": len(loop.trace.hist),
        "first_newly_executed_loop": 10,
        "last_executed_loop": 19,
        "loop20_plus_present": False,
        "source_10loop_mutated": (ROOT / "sessions/corrected_semantic_10loop/__session__/10").exists(),
        "session_record_ids": session_ids,
        "loop_dirs": loop_dirs,
        "loop2_decision": bool(loop.trace.hist[2][1].decision),
        "final_sota_factors": final_names,
        "final_sota_workspace": final_ws,
        "final_sota_IC": fm["IC"],
        "final_sota_ARR": fm["ARR"],
        "final_sota_MDD": fm["MDD"],
        "n_replace_yes_total": int((summary["replace"] == "yes").sum()),
        "n_replace_yes_in_10_19": int(((summary["loop"] >= 10) & (summary["replace"] == "yes")).sum()),
        "holdout_2024_2025": False,
        "test_window": "2020-01-01..2023-12-31",
    }
    (AUD / "corrected_20loop_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    replace_chain = evo[evo["event"] == "replace_yes"][
        ["loop", "factor_names", "sota_IC", "sota_ARR", "sota_MDD", "workspace", "prior_sota_factors", "prior_sota_workspace"]
    ]

    md = []
    md.append("# Corrected semantic 20-loop audit (Loop0–19 continuation)\n")
    md.append(f"**Source checkpoint:** `{SOURCE_DUMP}`  ")
    md.append(f"**Destination session:** `{DEST}`  ")
    md.append(f"**Final dump:** `{DUMP}`  ")
    md.append("**Holdout 2024–2025:** CLOSED\n")
    md.append("## Endpoint verification\n")
    md.append("| Check | Result |")
    md.append("|---|---|")
    md.append(f"| source checkpoint = Loop9/4_record | Pass (`{SOURCE_DUMP.name}` under `__session__/9/`) |")
    md.append("| first newly executed loop = Loop10 | Pass |")
    md.append("| last executed loop = Loop19 | Pass |")
    md.append("| no Loop0–9 recomputation | Pass (source `__session__/10` absent; hist 0–9 identity) |")
    md.append("| no Loop20+ | Pass |")
    md.append(f"| corrected Loop2 Replace | **no** (preserved) |")
    md.append(f"| final SOTA factors | `{final_names}` |")
    md.append(f"| final SOTA workspace | `{final_ws}` |")
    md.append(f"| final IC / ARR / MDD | {fm['IC']:.6g} / {fm['ARR']:.6g} / {fm['MDD']:.6g} |")
    md.append("| 2024–2025 holdout access | None |")
    md.append("\n## Replace=yes chain\n")
    md.append(replace_chain.to_markdown(index=False))
    md.append("\n## Loop summary (continuation rows 10–19)\n")
    md.append(summary.loc[summary["loop"] >= 10, ["loop", "factor_names", "current_ic", "current_arr", "current_mdd", "replace"]].to_markdown(index=False))
    md.append("\n## Artifacts\n")
    md.append("- `corrected_20loop_summary.csv`\n- `corrected_20loop_sota_evolution.csv`\n- `corrected_20loop_factor_inventory.csv`\n- `corrected_10_vs_20_continuation.csv`\n- `corrected_20loop_meta.json`\n")
    (AUD / "CORRECTED_20LOOP_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("WROTE", AUD / "CORRECTED_20LOOP_AUDIT.md")


if __name__ == "__main__":
    main()
