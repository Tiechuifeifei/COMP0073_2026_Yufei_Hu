#!/usr/bin/env python
"""Launch RD-Agent factor loop with 07-07 golden prompts overlaid (no RD-Agent tree writes).

Patches rdagent.utils.agent.tpl.load_content so scenarios.qlib.prompts loads from
reports/rdagent_matched_reference_rerun_20260916/golden_prompts/scenarios_qlib_prompts.yaml
instead of the worktree file that may still contain diversity §6.
"""
from __future__ import annotations
import os

import sys
from pathlib import Path


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


ROOT = (Path(os.environ["PROJECT_ROOT"]) / "reports/rdagent_matched_reference_rerun_20260916") if os.environ.get("PROJECT_ROOT") else (_repo_root() / "reports/rdagent_matched_reference_rerun_20260916")
GOLDEN_QLIB = ROOT / "golden_prompts" / "scenarios_qlib_prompts.yaml"
EXPECTED_SHA = "f8a213cb35c6677a9a4a38ccbdfb4b05fe4798090da83764b8fafe0f1dc98a1c"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_prompt_overlay() -> None:
    if not GOLDEN_QLIB.is_file():
        raise SystemExit(f"missing golden prompt: {GOLDEN_QLIB}")
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
            assert path_part == "scenarios.qlib.prompts"
            content = golden_yaml
            for key in (k for yt in yaml_trace for k in yt.split(".")):
                content = content[key]
            # fail closed if diversity leaked into golden
            blob = content if isinstance(content, str) else yaml.safe_dump(content)
            if "Maintain Exploration Diversity Across Factor Families" in blob:
                raise SystemExit("FAIL: diversity text found in golden overlay content")
            return content
        return orig(uri, caller_dir=caller_dir, ftype=ftype)

    tpl_mod.load_content = load_content  # type: ignore[assignment]
    print(f"PROMPT_OVERLAY active: {GOLDEN_QLIB} sha256={got}", flush=True)


def main() -> None:
    # Ensure RD-Agent is importable
    rd_root = Path(os.environ["RDAGENT_ROOT"])
    if str(rd_root) not in sys.path:
        sys.path.insert(0, str(rd_root))

    _install_prompt_overlay()

    # Import after patch so runtime T() uses overlay
    from rdagent.app.qlib_rd_loop.factor import main as factor_main

    # fire-compatible: pass through CLI args after this script name
    import fire

    fire.Fire(factor_main)


if __name__ == "__main__":
    main()
