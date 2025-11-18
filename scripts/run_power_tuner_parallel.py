from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import pandas as pd

sys.path.append(os.path.abspath('.'))

from power_tuner import (
    ORIGINAL_COVARS,
    run_search_parallel_from_path,
    analyze_results,
    write_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CCUS-informed Power Tuner in parallel (2–3 bins, VIF, R optional)")
    p.add_argument("--data", default="ds_no_prep.csv", help="Path to raw dataset CSV")
    p.add_argument("--artifact-dir", default="./_artifacts", help="Artifact output directory base")
    p.add_argument("--target-mde", type=float, default=0.015, help="Target MDE threshold")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    p.add_argument("--power", type=float, default=0.80, help="Power target for MDE")
    p.add_argument("--max-candidates", type=int, default=160, help="Max grid candidates to evaluate")
    p.add_argument("--threads", type=int, default=4, help="Parallel workers (processes)")
    p.add_argument("--use-wcb", action="store_true", help="Enable R-based WCB for pooled ATT and ES pretrend WCB")
    p.add_argument("--honestdid", action="store_true", help="Enable HonestDiD (R) for robustness flags")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build base StudyConfig kwargs (without df)
    base_kwargs: Dict[str, Any] = dict(
        artifact_dir=args.artifact_dir,
        covariates=list(ORIGINAL_COVARS),
        use_wcb=bool(args.use_wcb),
        honestdid_enable=bool(args.honestdid),
    )

    print(f"[PowerTuner] Parallel run: {args.max_candidates} candidates, threads={args.threads}")
    results_df, summary = run_search_parallel_from_path(
        data_path=args.data,
        base_config_kwargs=base_kwargs,
        target_mde=float(args.target_mde),
        threads=int(args.threads),
        alpha=float(args.alpha),
        power_target=float(args.power),
        max_candidates=int(args.max_candidates),
        use_wcb=bool(args.use_wcb),
        honestdid=bool(args.honestdid),
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

