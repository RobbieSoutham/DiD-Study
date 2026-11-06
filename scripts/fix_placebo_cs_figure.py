"""Render the Callaway-Sant'Anna placebo distribution under the shared theme.

Reads ``results/data/placebo_cs_draws.csv`` and
``results/data/placebo_cs_summary.json`` and writes both PNG and PDF versions
to ``results/figures`` and the Paper B figure folder.
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
    COL_REF,
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

    draws = pd.read_csv(DATA_DIR / "placebo_cs_draws.csv")
    with open(DATA_DIR / "placebo_cs_summary.json", "r", encoding="utf-8") as fh:
        summary = json.load(fh)

    vals = draws.loc[draws["status"].eq("ok"), "att_circ"].dropna().astype(float)
    observed = float(summary["observed_att_circ"])
    q025 = float(summary["placebo_q025"])
    q975 = float(summary["placebo_q975"])
    pvals = summary.get("p_values", {})
    upper_p = float(pvals.get("one_sided_upper", pvals.get("p_one_sided_upper")))
    two_p = float(pvals.get("two_sided_abs", pvals.get("p_two_sided_abs")))

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.hist(vals, bins=40, color="0.78", edgecolor="white")
    zero_line(ax, orient="v")
    ax.axvline(
        observed,
        color=COL_POOLED,
        linewidth=2.0,
        label=rf"Observed $\widehat{{ATT}}^\circ = {observed:.3f}$",
    )
    ax.axvline(q025, color=COL_REF, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axvline(
        q975,
        color=COL_REF,
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
        label=r"Placebo 95\% interval",
    )
    ax.set_xlabel(r"Placebo $\widehat{ATT}^{\circ}$")
    ax.set_ylabel("Number of draws")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    primary = FIG_DIR / "placebo_cs_distribution.png"
    primary_pdf = primary.with_suffix(".pdf")
    fig.savefig(primary, dpi=220, bbox_inches="tight")
    fig.savefig(primary_pdf, bbox_inches="tight")
    plt.close(fig)
    shutil.copyfile(primary, PAPER_FIG_DIR / primary.name)
    shutil.copyfile(primary_pdf, PAPER_FIG_DIR / primary_pdf.name)
    print(f"wrote {primary} / {primary_pdf} and copied both to paper figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
