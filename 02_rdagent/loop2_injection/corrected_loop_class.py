"""Picklable FactorRDLoop subclass for corrected Loop2 injection."""
from __future__ import annotations
import os

from pathlib import Path
from typing import Any

from rdagent.app.qlib_rd_loop.factor import FactorRDLoop


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
WS_ROOT = (Path(os.environ["RDAGENT_ROOT"]) / "git_ignore_folder/RD-Agent_workspace")
ORIG_ATR20 = WS_ROOT / "6e214d294dd345c3a11631513abce714" / "factor.py"
ORIG_VR20 = WS_ROOT / "84d96ef0edef4d7b8eada48557805d0f" / "factor.py"
CORRECTED_FACTOR_PY = ROOT / "factors" / "atr_20d_x_volume_ratio_20d_corrected_factor.py"


def _materialize_result_h5(ws, df) -> None:
    """Ensure result.h5 exists even when execute() is served from pickle cache."""
    out = ws.workspace_path / "result.h5"
    if out.exists():
        return
    if df is None:
        raise RuntimeError(f"execute returned no dataframe for {ws.target_task.factor_name}")
    ws.workspace_path.mkdir(parents=True, exist_ok=True)
    df.to_hdf(out, key="data", mode="w")


def build_corrected_loop2_experiment(trace):
    from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback, CoSTEERSingleFeedback
    from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
    from rdagent.core.proposal import Hypothesis
    from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
    from rdagent.utils.qlib import ALPHA20

    hypo = Hypothesis(
        hypothesis=(
            "We propose three new factors: 1) 20-day ATR (ATR_20d), 2) 20-day volume ratio "
            "(volume_ratio_20d), and 3) their multiplicative cross term (ATR_20d × volume_ratio_20d). "
            "CORRECTED IMPLEMENTATION: interaction uses relative classic ATR (TR/prev_close) times "
            "the original VR20_orig component (volume/rolling_mean(volume,20))."
        ),
        reason=(
            "Semantic correction of the Loop2 interaction only: remove absolute-price scale from ATR "
            "while keeping VR20_orig identical to the audited original interaction component."
        ),
        concise_reason="Correct Loop2 interaction scale (relative ATR x VR20_orig).",
        concise_observation="Original interaction scaled with price; corrected is scale-invariant.",
        concise_justification="Frozen ablation correction; VR and sibling factors unchanged.",
        concise_knowledge="Use rel_ATR20 * VR20_orig for atr_20d_x_volume_ratio_20d.",
    )

    tasks = [
        FactorTask(
            factor_name="atr_20d",
            factor_description="[Volatility Factor] 20-day average true range capturing longer-term range-based volatility signals.",
            factor_formulation=r"\mathrm{ATR}_{t}^{20} = \frac{1}{20} \sum_{i=0}^{19} \frac{h_{t-i} - \ell_{t-i}}{c_{t-i}}",
            variables={
                "h_{t-i}": "high price of the instrument on day t−i",
                "\\ell_{t-i}": "low price of the instrument on day t−i",
                "c_{t-i}": "closing price of the instrument on day t−i",
                "window_size": "look-back period for ATR calculation (20 days)",
            },
        ),
        FactorTask(
            factor_name="volume_ratio_20d",
            factor_description="[Liquidity Factor] 20-day volume ratio highlighting sustained liquidity breakouts.",
            factor_formulation=r"\mathrm{VolRatio}_{t}^{20} = \frac{v_{t}}{\frac{1}{20} \sum_{i=1}^{20} v_{t-i}} - 1",
            variables={
                "v_{t}": "trading volume of the instrument on day t",
                "v_{t-i}": "trading volume of the instrument on day t−i",
                "window_size": "look-back period for volume ratio calculation (20 days)",
            },
        ),
        FactorTask(
            factor_name="atr_20d_x_volume_ratio_20d",
            factor_description=(
                "[Interaction Factor] Cross term of 20-day relative ATR and original VR20_orig "
                "(semantic-corrected; scale-invariant)."
            ),
            factor_formulation=(
                r"\mathrm{relATR}_{t}^{20}\times\mathrm{VR}^{orig}_{t},"
                r"\quad \mathrm{relTR}_t=\max(H-L,|H-C_{prev}|,|L-C_{prev}|)/C_{prev}"
            ),
            variables={
                "relATR": "20-day mean of classic TR / prev_close",
                "VR_orig": "volume / rolling_mean(volume, 20) (same as original interaction)",
            },
        ),
    ]

    exp = QlibFactorExperiment(tasks, hypothesis=hypo)
    empty_baseline = QlibFactorExperiment(sub_tasks=[])
    accepted = [t[0] for t in trace.hist if t[1]]
    exp.based_experiments = [empty_baseline] + accepted
    exp.factor_library_experiments = None
    exp.base_features = dict(ALPHA20)
    exp.base_feature_codes = {}
    exp._is_corrected_loop2 = True  # type: ignore[attr-defined]

    codes = {
        "atr_20d": ORIG_ATR20.read_text(encoding="utf-8"),
        "volume_ratio_20d": ORIG_VR20.read_text(encoding="utf-8"),
        "atr_20d_x_volume_ratio_20d": CORRECTED_FACTOR_PY.read_text(encoding="utf-8"),
    }
    sub_ws = []
    coder_fbs = []
    for task in tasks:
        ws = FactorFBWorkspace(target_task=task)
        ws.inject_files(**{"factor.py": codes[task.factor_name]})
        # Use "All" to match process_factor_data / QlibFactorRunner re-execute hash.
        feedback, df = ws.execute(data_type="All")
        _materialize_result_h5(ws, df)
        print(
            f"factor {task.factor_name} execute: {str(feedback)[:120]} "
            f"path={ws.workspace_path} result.h5={(ws.workspace_path / 'result.h5').exists()} "
            f"shape={None if df is None else df.shape}",
            flush=True,
        )
        if not (ws.workspace_path / "result.h5").exists():
            raise RuntimeError(f"missing result.h5 for {task.factor_name} after execute")
        task.factor_implementation = True
        sub_ws.append(ws)
        coder_fbs.append(
            CoSTEERSingleFeedback(
                execution=str(feedback),
                return_checking="Pre-executed corrected Loop2 factor; value present.",
                code="Frozen original sibling / corrected interaction implementation accepted.",
                final_decision=True,
            )
        )
    exp.sub_workspace_list = sub_ws
    # Required by process_factor_data: zip(sub_workspace_list, prop_dev_feedback).
    exp.prop_dev_feedback = CoSTEERMultiFeedback(coder_fbs)
    return hypo, exp


class CorrectedSemanticFactorRDLoop(FactorRDLoop):
    """Picklable FactorRDLoop with one-shot corrected Loop2 injection."""

    corrected_loop2_injected: bool = False
    allow_corrected_injection: bool = True

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        if (
            self.allow_corrected_injection
            and (not getattr(self, "corrected_loop2_injected", False))
            and len(self.trace.hist) == 2
        ):
            print("Injecting corrected Loop2 experiment (native FactorRDLoop path)", flush=True)
            hypo, exp = build_corrected_loop2_experiment(self.trace)
            return {"propose": hypo, "exp_gen": exp}
        return await super().direct_exp_gen(prev_out)

    def coding(self, prev_out: dict[str, Any]):
        exp = prev_out["direct_exp_gen"]["exp_gen"]
        if getattr(exp, "_is_corrected_loop2", False):
            print("Coding step: using pre-executed corrected Loop2 workspaces", flush=True)
            self.corrected_loop2_injected = True
            from rdagent.log import rdagent_logger as logger

            logger.log_object(exp.sub_workspace_list, tag="coder result")
            return exp
        return super().coding(prev_out)
