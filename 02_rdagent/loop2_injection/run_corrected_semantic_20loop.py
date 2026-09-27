#!/usr/bin/env python3
"""Continue the corrected semantic RD8 branch from Loop 9 through Loop 19.

Source: sessions/corrected_semantic_10loop/__session__/9/4_record
Destination: sessions/corrected_semantic_20loop/
Loops 0–9 are reused as-is; Loop 2 is not re-injected here."""

from __future__ import annotations
import os

import asyncio
import json
import sys
from pathlib import Path
from typing import Any


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
SOURCE_SESSION = ROOT / "sessions" / "corrected_semantic_10loop"
LOOP9_DUMP = SOURCE_SESSION / "__session__" / "9" / "4_record"
DEST_SESSION = ROOT / "sessions" / "corrected_semantic_20loop"
GOLDEN_QLIB = ROOT / "golden_prompts" / "scenarios_qlib_prompts.yaml"
EXPECTED_SHA = "f8a213cb35c6677a9a4a38ccbdfb4b05fe4798090da83764b8fafe0f1dc98a1c"

EXPECTED_SOTA = {
    "factors": ["dynamic_price_momentum", "dynamic_volume_momentum", "rv_corr_20d"],
    "IC": 0.0060463724171109,
    "ARR": 0.1661358465623283,
    "MDD": -0.2515951252551563,
}

sys.path.insert(0, os.environ["RDAGENT_ROOT"])
sys.path.insert(0, str(ROOT / "scripts"))


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_prompt_overlay() -> None:
    got = _sha256(GOLDEN_QLIB)
    if got != EXPECTED_SHA:
        raise SystemExit(f"golden prompt hash drift: {got} != {EXPECTED_SHA}")
    import yaml
    from rdagent.utils.agent import tpl as tpl_mod

    golden_yaml = yaml.safe_load(GOLDEN_QLIB.read_text(encoding="utf-8"))
    orig = tpl_mod.load_content

    def load_content(uri: str, caller_dir=None, ftype: str = "yaml"):  # type: ignore[no-untyped-def]
        if ftype == "yaml" and uri.startswith("scenarios.qlib.prompts"):
            path_part, *yaml_trace = uri.split(":")
            content = golden_yaml
            for key in (k for yt in yaml_trace for k in yt.split(".")):
                content = content[key]
            blob = content if isinstance(content, str) else yaml.safe_dump(content)
            if "Maintain Exploration Diversity Across Factor Families" in blob:
                raise SystemExit("FAIL: diversity text in golden overlay")
            return content
        return orig(uri, caller_dir=caller_dir, ftype=ftype)

    tpl_mod.load_content = load_content  # type: ignore[assignment]
    print(f"PROMPT_OVERLAY active sha256={got}", flush=True)


def print_preflight(sota_names: list[str], ic: float, arr: float, mdd: float) -> None:
    print("=" * 72, flush=True)
    print("PREFLIGHT — corrected semantic Loop10-19 continuation", flush=True)
    print("=" * 72, flush=True)
    print(f"source_checkpoint: {LOOP9_DUMP}", flush=True)
    print(f"source_session_readonly: {SOURCE_SESSION}", flush=True)
    print(f"source_SOTA_factors: {'|'.join(sota_names)}", flush=True)
    print(f"source_SOTA_IC: {ic}", flush=True)
    print(f"source_SOTA_ARR: {arr}", flush=True)
    print(f"source_SOTA_MDD: {mdd}", flush=True)
    print(f"destination_session: {DEST_SESSION}", flush=True)
    print("first_newly_executed_loop: 10", flush=True)
    print("final_loop: 19", flush=True)
    print("loop_n_cli: 20  (target total from index 0; 0-9 no-op, 10-19 run)", flush=True)
    print("holdout_2024_2025: CLOSED (test window ends 2023-12-31)", flush=True)
    print("injection: DISABLED (native Loop10-19 only)", flush=True)
    print("recompute_loop0_9: NO", flush=True)
    print("=" * 72, flush=True)


async def run_pipeline() -> None:
    install_prompt_overlay()
    from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
    from rdagent.utils.qlib import ALPHA20

    from corrected_loop_class import CorrectedSemanticFactorRDLoop

    if not LOOP9_DUMP.exists():
        raise SystemExit(f"Missing Loop9 dump: {LOOP9_DUMP}")
    if not SOURCE_SESSION.exists():
        raise SystemExit(f"Missing source session: {SOURCE_SESSION}")

    if DEST_SESSION.exists():
        if (DEST_SESSION / "__session__" / "10").exists() or (DEST_SESSION / "Loop_10").exists():
            raise SystemExit(f"Refuse overwrite: Loop10+ already present in {DEST_SESSION}")
        if (DEST_SESSION / "__session__" / "19").exists() or (DEST_SESSION / "Loop_19").exists():
            raise SystemExit(f"Refuse overwrite: Loop19 already present in {DEST_SESSION}")

    # Source session must remain untouched — checkout to a different path.
    print(f"Loading Loop9 dump (read-only source): {LOOP9_DUMP}", flush=True)
    print(f"Checkout destination session: {DEST_SESSION}", flush=True)
    base = FactorRDLoop.load(str(LOOP9_DUMP), checkout=DEST_SESSION)

    # Ensure picklable corrected subclass; disable one-shot Loop2 injection.
    if not isinstance(base, CorrectedSemanticFactorRDLoop):
        base.__class__ = CorrectedSemanticFactorRDLoop
    base.allow_corrected_injection = False
    base.corrected_loop2_injected = True
    loop: Any = base

    assert len(loop.trace.hist) == 10, f"expected hist len 10, got {len(loop.trace.hist)}"
    # Corrected trajectory gates
    assert not bool(loop.trace.hist[2][1].decision), "corrected Loop2 must remain Replace=no"
    assert bool(loop.trace.hist[3][1].decision), "Loop3 must be Replace=yes"
    assert bool(loop.trace.hist[9][1].decision), "Loop9 must be Replace=yes"
    names2 = [t.factor_name for t in loop.trace.hist[2][0].sub_tasks]
    assert names2 == ["atr_20d", "volume_ratio_20d", "atr_20d_x_volume_ratio_20d"], names2

    sota = loop.trace.get_sota_experiment()
    assert sota is not None
    sota_names = [t.factor_name for t in sota.sub_tasks]
    ic = float(sota.result["IC"])
    arr = float(sota.result["1day.excess_return_with_cost.annualized_return"])
    mdd = float(sota.result["1day.excess_return_with_cost.max_drawdown"])
    assert sota_names == EXPECTED_SOTA["factors"], sota_names
    assert abs(ic - EXPECTED_SOTA["IC"]) < 1e-15
    assert abs(arr - EXPECTED_SOTA["ARR"]) < 1e-15
    assert abs(mdd - EXPECTED_SOTA["MDD"]) < 1e-15

    # Forbidden original principal Loop2 workspace must not be SOTA
    sota_ws = str(sota.experiment_workspace.workspace_path)
    for bad in ("61a0e3a7267949579b32513f34c3c930", "3f9b4b2805d7420a80cebd090d6a1afa"):
        assert bad not in sota_ws, f"forbidden principal workspace in SOTA: {bad}"

    print_preflight(sota_names, ic, arr, mdd)
    (ROOT / "gates" / "loop9_continue_parity_ok.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(LOOP9_DUMP),
                "destination_session": str(DEST_SESSION),
                "hist_len": 10,
                "sota_factors": sota_names,
                "IC": ic,
                "ARR": arr,
                "MDD": mdd,
                "loop2_decision": False,
                "loop3_decision": True,
                "loop9_decision": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loop.plan["features"] = dict(ALPHA20)
    loop.plan["feature_codes"] = {}

    print("Resuming Loop10-19 with loop_n=20 (kickoffs 0-9 no-op; 10-19 run)...", flush=True)
    await loop.run(loop_n=20)

    # Post-run hard checks
    assert len(loop.trace.hist) == 20, f"expected hist 20, got {len(loop.trace.hist)}"
    if (DEST_SESSION / "__session__" / "20").exists() or (DEST_SESSION / "Loop_20").exists():
        raise SystemExit("FAIL: Loop20+ present")
    if not (DEST_SESSION / "__session__" / "10").exists():
        raise SystemExit("FAIL: Loop10 session dump missing")
    if not (DEST_SESSION / "__session__" / "19").exists():
        raise SystemExit("FAIL: Loop19 session dump missing")
    # Source session must still stop at 9
    if (SOURCE_SESSION / "__session__" / "10").exists():
        raise SystemExit("FAIL: source 10loop session was mutated with Loop10+")

    print("Completed target Loop0-19 (Loop10-19 newly executed)", flush=True)
    (ROOT / "canonical_20loop_session_path.txt").write_text(str(DEST_SESSION) + "\n", encoding="utf-8")


def main() -> None:
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
