"""Rebuild only the Paper B R1.1 figures from already-computed CSV results.

This script is used by the figure-styling pass: it reads the persisted
CSVs in ``results/data/`` and re-renders the affected figures with the
shared style module so we do not have to re-run the full pipeline.

Figure files regenerated:
  - event_study.png         (from event_study_coefs.csv)
  - differences_att_gt.png  (from differences_event_agg.csv)
  - support_by_tau.png      (from support_by_tau.csv)
  - loo_cs_diagnostic.png   (from treated_units_revision_support.csv,
                             if results/data/loo.csv missing - otherwise
                             uses that)
  - dose_distribution.png   (Small/Medium/Large treated-unit-year counts from
                             treated_units_revision_support.csv)
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the package importable when this script is run directly.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from did_study.reporting.style import (  # noqa: E402
    COL_POOLED,
    COL_PRE,
    COL_REF,
    PALETTE_DOSE,
    apply_paper_theme,
    dose_colour,
    ref_line,
    zero_line,
)

DATA_DIR = REPO_ROOT / "results" / "data"
FIG_DIR = REPO_ROOT / "results" / "figures"
PAPER_FIG_DIR = (
    REPO_ROOT.parent.parent / "papers" / "b_ccus_did" / "figures"
).resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, name: str) -> None:
    """Save *fig* as PNG and PDF, then copy both to paper figures."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    primary = FIG_DIR / name
    primary_pdf = primary.with_suffix(".pdf")
    fig.savefig(primary, dpi=200, bbox_inches="tight")
    fig.savefig(primary_pdf, bbox_inches="tight")
    plt.close(fig)

    shutil.copyfile(primary, PAPER_FIG_DIR / name)
    shutil.copyfile(primary_pdf, PAPER_FIG_DIR / primary_pdf.name)
    print(
        f"  wrote {primary.relative_to(REPO_ROOT)} / "
        f"{primary_pdf.relative_to(REPO_ROOT)} and copied both"
    )


def _bin_for_capacity(cap: float) -> str:
    """Return the dose-bin label for a numeric initial capacity (Mt CO2/yr).

    Matches the [0, 1), [1, 5), [5, inf) bins used elsewhere in the pipeline.
    """
    if cap < 1.0:
        return "Small"
    if cap < 5.0:
        return "Medium"
    return "Large"


def _plot_split_event_trajectory(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    value_col: str,
) -> None:
    """Draw colour-consistent pre/post paths around the omitted reference.

    Event time ``tau=-1`` is normalised to zero by construction; it is not an
    estimated coefficient and therefore receives no confidence interval.  The
    open marker makes that distinction visible while preventing the previous
    pale-blue interpolation directly from ``tau=-2`` to ``tau=0``.
    """
    ordered = df.sort_values("event_time")
    pre = ordered[ordered["event_time"] < -1]
    post = ordered[ordered["event_time"] >= 0]

    if not pre.empty:
        pre_x = np.append(pre["event_time"].to_numpy(dtype=float), -1.0)
        pre_y = np.append(pre[value_col].to_numpy(dtype=float), 0.0)
        ax.plot(
            pre_x,
            pre_y,
            color=COL_PRE,
            alpha=0.78,
            linewidth=1.4,
            zorder=1,
        )

    if not post.empty:
        post_x = np.insert(post["event_time"].to_numpy(dtype=float), 0, -1.0)
        post_y = np.insert(post[value_col].to_numpy(dtype=float), 0, 0.0)
        ax.plot(
            post_x,
            post_y,
            color=COL_POOLED,
            alpha=0.78,
            linewidth=1.4,
            zorder=1,
        )

    ax.scatter(
        [-1.0],
        [0.0],
        s=34,
        facecolors="white",
        edgecolors=COL_REF,
        linewidths=1.1,
        zorder=4,
    )


# ---------------------------------------------------------------------------
# Figure 1 - event study
# ---------------------------------------------------------------------------


def figure_event_study() -> None:
    df = pd.read_csv(DATA_DIR / "event_study_coefs.csv")
    df = df.sort_values("event_time")

    if "lo" not in df or "hi" not in df:
        df["lo"] = df["beta"] - 1.96 * df["se"]
        df["hi"] = df["beta"] + 1.96 * df["se"]

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    # Split pre vs post so we can colour-code.
    pre_mask = df["event_time"] < 0
    post_mask = ~pre_mask

    for mask, colour, label in (
        (pre_mask, COL_PRE, "Pre-treatment"),
        (post_mask, COL_POOLED, "Post-treatment"),
    ):
        sub = df[mask]
        if sub.empty:
            continue
        x = sub["event_time"].to_numpy(dtype=float)
        y = sub["beta"].to_numpy(dtype=float)
        lo = sub["lo"].to_numpy(dtype=float)
        hi = sub["hi"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=[y - lo, hi - y],
            fmt="o",
            color=colour,
            ecolor=colour,
            capsize=3,
            linewidth=1.6,
            markersize=5,
            label=label,
        )

    # Preserve the pre/post colour semantics and show the omitted reference
    # as a zero-normalised anchor, not an estimated coefficient.
    _plot_split_event_trajectory(ax, df, value_col="beta")

    zero_line(ax, orient="h")
    ref_line(ax, tau=-1)

    # Integer ticks every 1 across full range so tau in [-5, -2] is visible.
    xmin = int(np.floor(df["event_time"].min()))
    xmax = int(np.ceil(df["event_time"].max()))
    ax.set_xticks(np.arange(xmin, xmax + 1, 1))
    ax.set_xlim(xmin - 0.5, xmax + 0.5)

    ax.set_xlabel(r"Event time ($\tau$)")
    ax.set_ylabel("Effect on log emissions")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    _save(fig, "event_study.png")


# ---------------------------------------------------------------------------
# Figure 2 - differences ATT(g,t) event aggregation
# ---------------------------------------------------------------------------


def figure_differences_att_gt() -> None:
    df = pd.read_csv(DATA_DIR / "differences_event_agg.csv")
    df = df.sort_values("event_time")

    # Crop to V0 visual range [-5, +10] and omit the reference period,
    # matching Figure 1's event-study convention.
    df = df[(df["event_time"] >= -5) & (df["event_time"] <= 10)].copy()
    df = df[df["event_time"] != -1].copy()
    df["lo"] = df["att"] - 1.96 * df["se"]
    df["hi"] = df["att"] + 1.96 * df["se"]

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    pre_mask = df["event_time"] < 0
    post_mask = ~pre_mask

    for mask, colour, label in (
        (pre_mask, COL_PRE, "Pre-treatment"),
        (post_mask, COL_POOLED, "Post-treatment"),
    ):
        sub = df[mask]
        if sub.empty:
            continue
        x = sub["event_time"].to_numpy(dtype=float)
        y = sub["att"].to_numpy(dtype=float)
        lo = sub["lo"].to_numpy(dtype=float)
        hi = sub["hi"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=[y - lo, hi - y],
            fmt="o",
            color=colour,
            ecolor=colour,
            capsize=3,
            linewidth=1.6,
            markersize=5,
            label=label,
        )

    # Preserve the pre/post colour semantics and show the omitted reference
    # as a zero-normalised anchor, not an estimated coefficient.
    _plot_split_event_trajectory(ax, df, value_col="att")

    zero_line(ax, orient="h")
    ref_line(ax, tau=-1)

    # V0 cadence: integer ticks every 2.
    ax.set_xticks(np.arange(-5, 11, 2))
    ax.set_xlim(-5.5, 10.5)

    ax.set_xlabel(r"Event time ($\tau$)")
    ax.set_ylabel("ATT(g, t)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    _save(fig, "differences_att_gt.png")


# ---------------------------------------------------------------------------
# Legacy support-by-tau figure (not included in the current manuscript)
# ---------------------------------------------------------------------------


def figure_support_by_tau() -> None:
    df = pd.read_csv(DATA_DIR / "support_by_tau.csv")
    df = df[df["event_time"] < 0].sort_values("event_time")

    colour = COL_POOLED

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.bar(df["event_time"], df["units"], color=colour, edgecolor="none")
    ax.set_xlabel(r"Event time ($\tau$)")
    ax.set_ylabel("Number of treated units")
    fig.tight_layout()
    _save(fig, "support_by_tau.png")


# ---------------------------------------------------------------------------
# Figure 3 - leave-one-treated-unit-out Callaway--Sant'Anna diagnostic
# ---------------------------------------------------------------------------


def figure_loo() -> None:
    loo_path = DATA_DIR / "loo.csv"

    if loo_path.exists():
        loo_df = pd.read_csv(loo_path).copy()
        # Sort by point estimate so the figure remains stable across runs.
        loo_df = loo_df.sort_values("att_overall").reset_index(drop=True)
        labels = loo_df["dropped_unit"].astype(str).tolist()
        est = loo_df["att_overall"].to_numpy(dtype=float)
        lo = loo_df["ci_low"].to_numpy(dtype=float)
        hi = loo_df["ci_high"].to_numpy(dtype=float)
    else:
        # Fall back to the treated-units table: we don't have per-drop
        # estimates here, but we can at least render the unit list with
        # neutral markers so the figure structure is correct.  This
        # branch should not fire if the pipeline has been run, but we
        # keep it so the script is robust.
        treated = pd.read_csv(DATA_DIR / "treated_units_revision_support.csv")
        labels = [s.replace("--", " x ") for s in treated["country_sector"].astype(str)]
        est = np.zeros(len(labels))
        lo = np.full(len(labels), np.nan)
        hi = np.full(len(labels), np.nan)

    # Clean up labels: replace " -- " with " x " to match V0-style labels.
    labels = [s.replace("--", " x ") for s in labels]

    n = len(labels)
    height = max(3.5, 0.45 * n + 1.5)
    fig, ax = plt.subplots(figsize=(9.0, height))

    y = np.arange(n)
    zero_line(ax, orient="v")
    full_sample_att = 0.190
    ax.axvline(
        full_sample_att,
        color=COL_REF,
        linestyle=":",
        linewidth=1.5,
        alpha=0.85,
        zorder=0,
        label=r"Full-sample ATT$^{\circ}=0.190$",
    )
    have_ci = np.isfinite(lo).all() and np.isfinite(hi).all()
    if have_ci:
        xerr = np.vstack([est - lo, hi - est])
    else:
        xerr = None
    ax.errorbar(
        est,
        y,
        xerr=xerr,
        fmt="o",
        color=COL_POOLED,
        ecolor=COL_POOLED,
        capsize=3,
        linewidth=1.6,
        markersize=6,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"ATT$^{\circ}$ after dropping one treated unit")
    ax.set_ylabel("Dropped treated unit")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        frameon=False,
    )
    # No title (spec: remove plot title).
    fig.tight_layout()
    _save(fig, "loo_cs_diagnostic.png")


# ---------------------------------------------------------------------------
# dose_distribution
# ---------------------------------------------------------------------------


def figure_dose_distribution() -> None:
    """Three-bin chart of treated-unit-year initial capacities.

    This uses the corrected initial-capacity convention but aggregates directly
    to the descriptive Small/Medium/Large bins so the figure does not imply a
    rich within-bin continuous dose distribution from seven treated units.
    """
    treated = pd.read_csv(DATA_DIR / "treated_units_revision_support.csv")

    treated["dose_bin"] = treated["initial_capacity_mtco2yr"].astype(float).map(_bin_for_capacity)
    order = ["Small", "Medium", "Large"]
    counts = (
        treated.groupby("dose_bin", observed=True)["post_years"]
        .sum()
        .reindex(order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    x = np.arange(len(order))
    tick_labels = ["Small\n(<1)", "Medium\n(1-<5)", "Large\n(>=5)"]
    bars = ax.bar(
        x,
        counts.to_numpy(dtype=float),
        color=[dose_colour(label) for label in order],
        edgecolor="white",
        linewidth=0.8,
        width=0.72,
    )

    ax.bar_label(
        bars,
        labels=[str(int(v)) for v in counts],
        padding=4,
        fontsize=plt.rcParams["xtick.labelsize"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(r"Initial EOR--CCUS capacity bin (Mt CO$_2$/yr)")
    ax.set_ylabel(r"Number of treated unit--years")
    ax.set_ylim(0, max(float(counts.max()) * 1.18, 1.0))
    ax.margins(x=0.08)
    fig.tight_layout()
    _save(fig, "dose_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    apply_paper_theme()

    print("[1/5] event_study.png")
    figure_event_study()

    print("[2/5] differences_att_gt.png")
    figure_differences_att_gt()

    print("[3/5] support_by_tau.png")
    figure_support_by_tau()

    print("[4/5] loo_cs_diagnostic.png")
    figure_loo()

    print("[5/5] dose_distribution.png")
    figure_dose_distribution()

    print("\nDone.  Figures saved to:")
    print(f"  {FIG_DIR}")
    print(f"  {PAPER_FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
