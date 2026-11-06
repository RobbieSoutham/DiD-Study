"""Run only the leave-one-treated-unit-out diagnostic and persist results.

This re-uses the same StudyConfig as ``run_study_ccus.py`` but skips the
event-study and HonestDiD work, so the seven LOO fits run as fast as
possible.  Outputs:

  results/data/loo.csv                       (per-drop ATT^o and CI)
  results/figures/leave_one_treated_unit_out.png   (legacy filename used
                                                   by _plot_loo)

The figure-styling pass uses the CSV via
``scripts/rebuild_paper_figures.py`` to produce ``loo_twfe_diagnostic.png``.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# Re-use the helpers defined in run_study_ccus.py.
import run_study_ccus as study_mod  # noqa: E402

from did_study.study import DidStudy  # noqa: E402


def main() -> int:
    print("[1/3] Loading raw panel ds_no_prep.csv ...")
    raw_df = pd.read_csv("ds_no_prep.csv")
    print(f"       Loaded {len(raw_df):,} rows.")

    print("[2/3] Running primary study (needed for treated-unit list) ...")
    cfg = study_mod._build_config(raw_df, honestdid_enable=False)
    primary = DidStudy(cfg).run(run_event_study=False, run_honest_did=False)
    panel_df = getattr(getattr(primary, "data", None), "panel", None)
    if panel_df is None:
        panel_df = getattr(primary, "panel_df", None)
    if panel_df is None:
        print("       [error] could not recover panel_df")
        return 1

    treated_units = study_mod._treated_unit_table(panel_df)
    print(f"       Identified {len(treated_units)} treated units.")

    print("[3/3] Running LOO ...")
    study_mod._run_leave_one_out(raw_df, treated_units)
    print("Done.  loo.csv and leave_one_treated_unit_out.png written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
