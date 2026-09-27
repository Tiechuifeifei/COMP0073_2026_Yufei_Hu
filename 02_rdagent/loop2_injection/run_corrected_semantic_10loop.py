#!/usr/bin/env python3
"""Launch corrected semantic 10-loop rerun from Loop1 checkpoint."""
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
PRINCIPAL = (Path(os.environ["PROJECT_ROOT"]) / "reports/rdagent_matched_reference_rerun_20260916") if os.environ.get("PROJECT_ROOT") else (_repo_root() / "reports/rdagent_matched_reference_rerun_20260916")
LOOP1_DUMP = PRINCIPAL / "sessions/matched_20loop_20260916_090024/__session__/1/4_record"
GOLDEN_QLIB = ROOT / "golden_prompts" / "scenarios_qlib_prompts.yaml"
EXPECTED_SHA = "f8a213cb35c6677a9a4a38ccbdfb4b05fe4798090da83764b8fafe0f1dc98a1c"

# Make scripts/ importable as a package path
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


def verify_pre_loop3_state(loop) -> None:
    hist = loop.trace.hist
    assert len(hist) == 3, f"expected hist len 3 (L0,L1,corrected L2), got {len(hist)}"
    names1 = [t.factor_name for t in hist[1][0].sub_tasks]
    names2 = [t.factor_name for t in hist[2][0].sub_tasks]
    assert names1 == ["reversal_1d", "atr_10d", "volume_ratio_10d"], names1
    assert names2 == ["atr_20d", "volume_ratio_20d", "atr_20d_x_volume_ratio_20d"], names2
    ws2 = str(hist[2][0].experiment_workspace.workspace_path)
    assert "61a0e3a7267949579b32513f34c3c930" not in ws2, "original Loop2 workspace leaked"
    for i, (exp, fb) in enumerate(hist):
        for be in getattr(exp, "based_experiments", []) or []:
            p = str(getattr(getattr(be, "experiment_workspace", None), "workspace_path", ""))
            assert "61a0e3a7267949579b32513f34c3c930" not in p, f"Loop2 SOTA leaked into based_exp hist[{i}]"
    sota = loop.trace.get_sota_experiment()
    sota_ws = str(sota.experiment_workspace.workspace_path)
    for bad in ("3f9b4b2805d7420a80cebd090d6a1afa", "61a0e3a7267949579b32513f34c3c930"):
        assert bad not in sota_ws, f"forbidden workspace in SOTA: {bad}"
    if hist[2][1].decision:
        inter_ws = hist[2][0].sub_workspace_list[2]
        code = (inter_ws.workspace_path / "factor.py").read_text(encoding="utf-8")
        assert "prev_close" in code and ("rel_tr" in code or "true_range / prev_close" in code.replace(" ", ""))
    payload = {
        "hist_len": len(hist),
        "loop1_factors": names1,
        "loop2_factors": names2,
        "loop2_decision": bool(hist[2][1].decision),
        "sota_workspace": sota_ws,
        "loop2_ic": float(hist[2][0].result["IC"]) if hist[2][0].result is not None else None,
        "loop2_arr": float(hist[2][0].result["1day.excess_return_with_cost.annualized_return"])
        if hist[2][0].result is not None
        else None,
    }
    (ROOT / "gates" / "pre_loop3_state_ok.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("PRE_LOOP3_STATE_OK", json.dumps(payload), flush=True)


async def run_pipeline() -> None:
    install_prompt_overlay()
    from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
    from rdagent.utils.qlib import ALPHA20

    from corrected_loop_class import CorrectedSemanticFactorRDLoop

    session = ROOT / "sessions" / "corrected_semantic_10loop"
    if session.exists() and (session / "__session__" / "2").exists():
        raise SystemExit(f"Refuse overwrite existing Loop2+ session: {session}")

    print(f"Loading Loop1 dump: {LOOP1_DUMP}", flush=True)
    print(f"Checkout new session: {session}", flush=True)
    base = FactorRDLoop.load(str(LOOP1_DUMP), checkout=session)
    base.__class__ = CorrectedSemanticFactorRDLoop
    base.corrected_loop2_injected = False
    base.allow_corrected_injection = True
    loop: Any = base

    sota = loop.trace.get_sota_experiment()
    assert sota is not None
    ic = float(sota.result["IC"])
    arr = float(sota.result["1day.excess_return_with_cost.annualized_return"])
    mdd = float(sota.result["1day.excess_return_with_cost.max_drawdown"])
    print(f"Loop1 SOTA parity IC={ic} ARR={arr} MDD={mdd}", flush=True)
    assert abs(ic - 1.002098552893985e-05) < 1e-15
    assert abs(arr - 0.0395424743714626) < 1e-15
    assert abs(mdd - (-0.3337506904335999)) < 1e-15
    assert len(loop.trace.hist) == 2
    (ROOT / "gates" / "loop1_parity_ok.json").write_text(
        json.dumps({"IC": ic, "ARR": arr, "MDD": mdd, "hist_len": 2}, indent=2),
        encoding="utf-8",
    )

    loop.plan["features"] = dict(ALPHA20)
    loop.plan["feature_codes"] = {}

    print("Running through corrected Loop2 (loop_n=3 kickoffs 0-2)...", flush=True)
    await loop.run(loop_n=3)
    verify_pre_loop3_state(loop)

    loop.allow_corrected_injection = False
    print("Resuming Loop3-9 with loop_n=10 (kickoffs 0-9; 0-2 no-op)...", flush=True)
    await loop.run(loop_n=10)
    print("Completed target Loop0-9", flush=True)
    (ROOT / "canonical_session_path.txt").write_text(str(session) + "\n", encoding="utf-8")


def main() -> None:
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
