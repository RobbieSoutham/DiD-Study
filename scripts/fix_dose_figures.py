"""Red-team fixes for dose_distribution and mean_dose_by_tau.

* dose_distribution.png: three aggregate bars for the descriptive Small/Medium/Large
  bins. This keeps the corrected initial-capacity convention while avoiding
  a visually sparse sub-bin histogram from seven treated units.

* mean_dose_by_tau.png: truncate the x-axis to tau in [-5, +10] (was extending to
  tau ~30, which the red-team flagged as out-of-estimation visual
  leverage). The Large bin has one Brazil--NGP/LNG unit, so the caption
  discloses that the series is a single-unit profile. Use the shared
  style helpers: grey-dashed zero, black-plain tau = -1.

Outputs both to ``results/figures/`` and copies into
``papers/b_ccus_did/figures/``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Skip expensive pipeline ops before importing pipeline modules.
os.environ.setdefault("CCUS_SKIP_LOO", "1")
os.environ.setdefault("CCUS_SKIP_WCB", "1")
os.environ.setdefault("CCUS_SKIP_HONESTDID", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from did_study.reporting.style import (  # noqa: E402
    COL_POOLED,
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


def _save(fig: plt.Figure, name: str) -> None:
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
    if cap < 1.0:
        return "Small"
    if cap < 5.0:
        return "Medium"
    return "Large"


# ---------------------------------------------------------------------------
# dose_distribution - three aggregate bins
# ---------------------------------------------------------------------------
def figure_dose_distribution_bins() -> None:
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
    tick_labels = [
        "Small\n(<1)",
        "Medium\n(1-<5)",
        "Large\n(>=5)",
    ]
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
# mean_dose_by_tau - mean dose by event time, truncated to estimation window
# ---------------------------------------------------------------------------
def figure_mean_dose_by_tau_truncated() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from run_study_ccus import _build_config  # noqa: E402
    from did_study.study import DidStudy  # noqa: E402

    raw_df = pd.read_csv(REPO_ROOT / "ds_no_prep.csv")
    cfg = _build_config(raw_df, honestdid_enable=False)
    result = DidStudy(cfg).run(run_event_study=False, run_honest_did=False)
    panel = getattr(getattr(result, "data", None), "panel", None)
    if panel is None:
        raise RuntimeError("Could not retrieve prepared panel.")

    # Recompute event time and per-unit dose bin to match the original plot.
    dfp = panel.copy()
    if "g" not in dfp.columns:
        g = dfp.loc[dfp["dose_level"] > 0].groupby("unit_id")["Year"].min()
        dfp["g"] = dfp["unit_id"].map(g)
    dfp["event_time"] = np.where(dfp["g"].notna(), dfp["Year"] - dfp["g"], np.nan)
    init = dfp.loc[dfp["Year"] == dfp["g"]].groupby("unit_id")["dose_level"].first()
    bin_series = pd.cut(
        init,
        [-np.inf, 1.0, 5.0, np.inf],
        labels=["Small", "Medium", "Large"],
        right=False,
    )
    dfp["dose_bin"] = dfp["unit_id"].map(bin_series)

    # Restrict to the estimation window tau in [-5, +10].
    dfp = dfp[dfp["event_time"].between(-5, 10)]
    dfp = dfp.dropna(subset=["dose_level", "event_time"])

    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    # Pooled series (deep blue).
    pooled = dfp.groupby("event_time")["dose_level"].agg(["mean", "std", "count"]).reset_index()
    pooled = pooled.sort_values("event_time")
    pooled["std"] = pooled["std"].fillna(0.0)
    pooled["sem"] = pooled["std"] / np.sqrt(pooled["count"].clip(lower=2))
    pooled["ci_low"] = np.where(
        pooled["count"] >= 2,
        np.maximum(0.0, pooled["mean"] - 1.96 * pooled["sem"]),
        pooled["mean"],
    )
    pooled["ci_high"] = np.where(
        pooled["count"] >= 2,
        pooled["mean"] + 1.96 * pooled["sem"],
        pooled["mean"],
    )

    ax.plot(
        pooled["event_time"], pooled["mean"], "-o", color=COL_POOLED,
        linewidth=2.0, markersize=5, label="Pooled", zorder=3,
    )
    for _, r in pooled.iterrows():
        ax.errorbar(
            [r["event_time"]],
            [r["mean"]],
            yerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]],
            fmt="none",
            color=COL_POOLED,
            capsize=3,
            linewidth=1.2,
            alpha=0.9,
        )

    # Per-bin series for all three dose bins. Large is n = 1, so it is a
    # single-unit profile rather than a within-bin average.
    for label in ("Small", "Medium", "Large"):
        sub = dfp[dfp["dose_bin"] == label]
        if sub.empty:
            continue
        g = sub.groupby("event_time")["dose_level"].agg(["mean", "std", "count"]).reset_index()
        g = g.sort_values("event_time")
        ax.plot(
            g["event_time"], g["mean"], "--o",
            color=PALETTE_DOSE[label], linewidth=1.4, markersize=4,
            label=label, alpha=0.9,
        )

    zero_line(ax, orient="h")
    ref_line(ax, tau=-1)
    ax.set_xlabel(r"Event time ($\tau$)")
    ax.set_ylabel(r"Mean operational capacity (Mt CO$_2$/yr)")
    ax.set_xlim(-5.5, 10.5)
    ax.set_xticks(np.arange(-5, 11, 1))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False, title="Dose bin")
    fig.tight_layout()
    _save(fig, "mean_dose_by_tau.png")


if __name__ == "__main__":
    apply_paper_theme()
    print("[1/2] dose_distribution.png (three aggregate bins) ...")
    figure_dose_distribution_bins()
    print("[2/2] mean_dose_by_tau.png (tau in [-5, 10]) ...")
    figure_mean_dose_by_tau_truncated()
    print("Done.")
