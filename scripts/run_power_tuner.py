from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List

import pandas as pd

sys.path.append(os.path.abspath('.'))

from did_study.helpers.config import StudyConfig
from power_tuner import (
    ORIGINAL_COVARS,
    run_search,
    analyze_results,
    write_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CCUS-informed Power Tuner (2–3 bins, VIF-screened)")
    p.add_argument("--data", default="ds_no_prep.csv", help="Path to raw dataset CSV")
    p.add_argument("--artifact-dir", default="./_artifacts", help="Artifact output directory base")
    p.add_argument("--target-mde", type=float, default=0.015, help="Target MDE threshold")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    p.add_argument("--power", type=float, default=0.80, help="Power target for MDE")
    p.add_argument("--max-candidates", type=int, default=48, help="Max grid candidates to evaluate")
    p.add_argument("--use-wcb", action="store_true", help="Enable WCB in base runs (may require R)")
    p.add_argument("--honestdid", action="store_true", help="Enable HonestDiD in base runs (requires R)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[PowerTuner] Loading data: {args.data}")
    df = pd.read_csv(args.data)

    cfg = StudyConfig(
        df=df,
        artifact_dir=args.artifact_dir,
        covariates=list(ORIGINAL_COVARS),
        use_wcb=bool(args.use_wcb),
        honestdid_enable=bool(args.honestdid),
    )

    print("[PowerTuner] Running grid search (2–3 bins only, VIF-screened covariates)...")
    results_df, summary = run_search(
        cfg,
        target_mde=float(args.target_mde),
        alpha=float(args.alpha),
        power_target=float(args.power),
        max_candidates=int(args.max_candidates),
    )

    out_dir = summary["out_dir"]
    print(f"[PowerTuner] Artifacts -> {out_dir}")
    print("[PowerTuner] Analyzing drivers of MDE/SE ...")
    drivers = analyze_results(results_df, out_dir)
    print("[PowerTuner] Writing report ...")
    write_report(results_df, summary, target_mde=float(args.target_mde), out_dir=out_dir, alpha=float(args.alpha))
    print("[PowerTuner] Done.")


if __name__ == "__main__":
    main()

