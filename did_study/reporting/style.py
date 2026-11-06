"""Shared figure-style helpers for the CCUS Paper B figures.

Colour and line conventions, applied consistently across every figure in
the manuscript:

* ``COL_POOLED`` (deep blue) for the pooled ATT^o, post-treatment markers,
  and any "observed" / point-estimate annotation (e.g. the observed
  within-transform coefficient on the placebo histogram).
* ``COL_PRE`` (muted red) for pre-treatment markers (event study, ATT(g,t)
  pre-period markers).
* ``COL_ZERO`` (grey) drawn as a *dashed* line marks the zero-effect /
  null reference (horizontal on coefficient plots, vertical at zero on
  histograms).
* ``COL_REF`` (black) drawn as a *plain* solid line marks the reference
  period :math:`\\tau=-1` on event-study-style plots.
* ``PALETTE_DOSE`` for the Small / Medium / Large dose-bin colour scheme,
  applied consistently across Figs 3, 5, 6, and any sibling figures.
  Chosen to be visually distinct from both ``COL_POOLED`` (blue) and
  ``COL_PRE`` (red) so the conventions above remain unambiguous when the
  dose palette appears in the same plot.

All figure-generation code in the paper should import these constants and
call :func:`apply_paper_theme` once at the top of its main entry point so
the seaborn theme is consistent across plots.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns


# Pooled / post-treatment / observed-coefficient colour (deep blue).
COL_POOLED: str = "#1f4e79"

# Pre-treatment marker colour (muted red).
COL_PRE: str = "#9b2f2f"

# Zero-effect reference line (grey, dashed) — matches PlotTheme zero_line
# (color "0.25", linestyle "--", alpha 0.6, linewidth 1) in plotting.py.
COL_ZERO: str = "0.25"

# Reference-period (\tau = -1) line (black, plain).
COL_REF: str = "#000000"

# Sequential teal -> orange -> purple ramp for Small / Medium / Large
# dose bins.  Chosen to be print-safe and reasonably colour-blind friendly,
# and visually distinct from the pre-treatment red and pooled blue used
# elsewhere in the same plots.
PALETTE_DOSE: Dict[str, str] = {
    "Small": "#2ca58d",   # teal
    "Medium": "#e09f3e",  # warm amber/orange
    "Large": "#6a4c93",   # muted purple
}


def apply_paper_theme(context: str = "paper") -> None:
    """Apply the original CCUS plotting theme used by ``PlotTheme``.

    Keep the paper-specific colour conventions in this module, but align
    the typography and grid with ``did_study.reporting.plotting.PlotTheme``:
    20 pt titles, 16 pt axis labels, 12 pt ticks/legend, 120 dpi figures,
    and a dashed grid with alpha 0.3.  This prevents the standalone
    rebuild scripts from silently reverting to seaborn's smaller qualitative
    defaults.
    """
    sns.set_theme(style="whitegrid", context=context)
    plt.rcParams.update(
        {
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 12,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "grid.linestyle": "--",
            "grid.alpha": 0.3,
        }
    )


def dose_colour(bin_label: str) -> str:
    """Return the palette colour for a Small/Medium/Large bin label.

    Falls back to the Large colour if the label is not one of the known bins.
    """
    return PALETTE_DOSE.get(str(bin_label), PALETTE_DOSE["Large"])


def zero_line(ax: plt.Axes, *, orient: str = "h", value: float = 0.0,
              label: Optional[str] = None) -> None:
    """Draw the canonical zero-effect reference line on *ax*.

    Grey dashed, drawn behind the data (zorder=0).  ``orient='h'`` draws a
    horizontal line at *value* (the default — for coefficient plots);
    ``orient='v'`` draws a vertical line at *value* (for histograms).
    """
    fn = ax.axhline if orient == "h" else ax.axvline
    fn(value, color=COL_ZERO, linestyle="--", linewidth=1.0,
       alpha=0.6, zorder=0, label=label)


def ref_line(ax: plt.Axes, *, tau: float = -1.0,
             label: Optional[str] = None) -> None:
    """Draw the canonical reference-period (\\tau = -1) line on *ax*.

    Black, plain (solid), drawn behind the data.  Used on event-study and
    event-time-axis figures to mark the omitted reference period.
    """
    ax.axvline(tau, color=COL_REF, linestyle="-", linewidth=1.0,
               alpha=0.85, zorder=0, label=label)
