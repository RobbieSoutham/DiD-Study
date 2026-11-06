"""Regenerate Appendix Figure A.1 (HonestDiD M-curve) under the shared
paper theme and colour conventions.

Reads ``results/data/honest_did_bounds.csv`` (the persisted bounds from
the main pipeline run) and renders:

* lower bound in the Small/Medium dose-palette colour (teal),
* upper bound in the Medium dose-palette colour (amber),
* identified-set fill between the two,
* :math:`\\hat{\\theta}` baseline overlay in pooled blue (dotted),
* grey-dashed horizontal zero reference,
* black-plain vertical M* line where 0 first enters the identified set.

Writes PNG and PDF copies to ``results/figures`` and
``papers/b_ccus_did/figures``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from did_study.reporting.style import (  # noqa: E402
    COL_POOLED,
    COL_REF,
    PALETTE_DOSE,
    apply_paper_theme,
    zero_line,
)

DATA_DIR = REPO_ROOT / "results" / "data"
FIG_DIR = REPO_ROOT / "results" / "figures"
PAPER_FIG_DIR = (
    REPO_ROOT.parent.parent / "papers" / "b_ccus_did" / "figures"
).resolve()

THETA_HAT = -0.198  # exposure-weighted dynamic effect, baseline (M = 0).


def main() -> int:
    apply_paper_theme()

    df = pd.read_csv(DATA_DIR / "honest_did_bounds.csv").sort_values("M")
    m = df["M"].to_numpy(dtype=float)
    lo = df["lo"].to_numpy(dtype=float)
    hi = df["hi"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    ax.fill_between(m, lo, hi, color=PALETTE_DOSE["Medium"], alpha=0.18, linewidth=0)
    ax.plot(m, lo, color=PALETTE_DOSE["Small"], linewidth=2.0, label="Lower bound")
    ax.plot(m, hi, color=PALETTE_DOSE["Medium"], linewidth=2.0, label="Upper bound")

    zero_line(ax, orient="h")
    ax.axhline(THETA_HAT, color=COL_POOLED, linewidth=2.0, linestyle=":",
               label=r"$\hat{\theta} = -0.198$")

    # M*: smallest M where 0 first enters [lo, hi].
    inside = (lo <= 0) & (0 <= hi)
    if inside.any():
        idx = int(np.argmax(inside))
        ax.axvline(m[idx], color=COL_REF, linewidth=1.0, linestyle="-",
                   alpha=0.85, zorder=0, label=f"$M^*\\approx{m[idx]:.2g}$")

    ax.set_xlabel(r"$M$ (relative magnitude bound)")
    ax.set_ylabel("Bounded average effect")
    # Legend lives below the figure (two-row legend, frameless) so it
    # cannot collide with the M* vertical at upper-left or with the
    # upper-bound trace at upper-right.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
        handletextpad=0.6,
        columnspacing=1.4,
    )
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    primary = FIG_DIR / "honest_did.png"
    primary_pdf = primary.with_suffix(".pdf")
    fig.savefig(primary, dpi=200, bbox_inches="tight")
    fig.savefig(primary_pdf, bbox_inches="tight")
    plt.close(fig)
    shutil.copyfile(primary, PAPER_FIG_DIR / "honest_did.png")
    shutil.copyfile(primary_pdf, PAPER_FIG_DIR / primary_pdf.name)
    print(f"wrote {primary} / {primary_pdf} and copied both to paper figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
