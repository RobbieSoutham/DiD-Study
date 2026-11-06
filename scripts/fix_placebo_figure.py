"""Re-render the placebo distribution figure from cached data.

Reads ``results/data/placebo_twfe_distribution.csv`` and
``results/data/placebo_twfe_summary.json`` (both written by
``run_placebo_and_balance.py``) and re-renders the histogram under the
shared paper theme with the new colour conventions:

* placebo-coefficient histogram bars: neutral grey,
* observed-coefficient marker: pooled blue (vertical line),
* zero reference: grey dashed.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from did_study.reporting.style import (  # noqa: E402
    COL_POOLED,
    apply_paper_theme,
    zero_line,
)

DATA_DIR = REPO_ROOT / "results" / "data"
FIG_DIR = REPO_ROOT / "results" / "figures"
PAPER_FIG_DIR = (
    REPO_ROOT.parent.parent / "papers" / "b_ccus_did" / "figures"
).resolve()


def main() -> int:
    apply_paper_theme()

    df = pd.read_csv(DATA_DIR / "placebo_twfe_distribution.csv")
    with open(DATA_DIR / "placebo_twfe_summary.json", "r", encoding="utf-8") as fh:
        summary = json.load(fh)

    placebo_vals = df.loc[df["kind"] == "placebo", "att_twfe"].to_numpy()
    observed = float(summary["observed_twfe"])
    upper_p = float(summary["one_sided_upper_p"])
    two_sided_p = float(summary["two_sided_empirical_p"])

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.hist(placebo_vals, bins=40, color="0.78", edgecolor="white")
    zero_line(ax, orient="v")
    ax.axvline(
        observed,
        color=COL_POOLED,
        linewidth=2,
        label=f"Observed TWFE = {observed:.3f}",
    )
    ax.set_xlabel("Placebo within-transform coefficient")
    ax.set_ylabel("Number of draws")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    primary = FIG_DIR / "placebo_twfe_distribution.png"
    primary_pdf = primary.with_suffix(".pdf")
    fig.savefig(primary, dpi=220, bbox_inches="tight")
    fig.savefig(primary_pdf, bbox_inches="tight")
    plt.close(fig)
    shutil.copyfile(primary, PAPER_FIG_DIR / "placebo_twfe_distribution.png")
    shutil.copyfile(primary_pdf, PAPER_FIG_DIR / primary_pdf.name)
    print(f"wrote {primary} / {primary_pdf} and copied both to paper figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
