
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union, Dict, Any

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from did_study.helpers.utils import tidy_differences_event_agg_df
from ..estimators.att_differences import DifferencesAttResult


# ================================
# Theme + Figure Finalizer
# ================================

# ----------------------------
# Global color policy (single source of truth)
# ----------------------------
DOSE_BIN_ORDER = ["Small", "Medium", "Large"]

def _canonical_bin_label(v: Any) -> str:
    """Return canonical 'Small'/'Medium'/'Large' based on string contents."""
    s = str(v).strip().lower()
    if "small" in s:
        return "Small"
    if "medium" in s:
        return "Medium"
    if "large" in s:
        return "Large"
    return str(v)

def _dose_color_map_from_palette(palette: Sequence[str]) -> dict:
    """
    Use theme palette consistently:
      pooled = palette[0] (blue)
      Small  = palette[1] (emerald)
      Medium = palette[2] (amber)
      Large  = palette[4] (violet)
    """
    return {
        "Pooled": palette[0],
        "Small": palette[1] if len(palette) > 1 else palette[0],
        "Medium": palette[2] if len(palette) > 2 else palette[0],
        "Large": palette[4] if len(palette) > 4 else palette[0],
    }


@dataclass
class PlotTheme:
    """Global plotting theme used by FigFinalizer.
    Keep all aesthetic knobs here so individual plotting functions
    deal ONLY with the data drawing (artists) and not with styling.
    """
    figsize: Tuple[float, float] = (10.0, 6.0)
    dpi: int = 120

    # Fonts / sizing
    title_size: int = 20
    label_size: int = 16
    tick_size: int = 12
    legend_size: int = 12
    legend_title: Optional[str] = None
    group_labels: Optional[Sequence[str]] = None

    # Lines / grid
    grid: bool = True
    grid_style: str = "--"
    grid_alpha: float = 0.3

    # Reference lines
    zero_line: bool = True           # add y=0 horizontal
    ref_event_line: Optional[float] = -1  # for ES; set to None to disable
    ref_event_style: str = "-"
    ref_event_alpha: float = 0.6

    # Layout
    tight_layout: bool = True
    constrained_layout: bool = False

    # Colors
    palette: Sequence[str] = field(default_factory=lambda: [
        "#2563eb",  # blue
        "#10b981",  # emerald
        "#f59e0b",  # amber
        "#ef4444",  # red
        "#8b5cf6",  # violet
        "#14b8a6",  # teal
        "#84cc16",  # lime
    ])


class FigFinalizer:
    """A decorator-like wrapper that centralizes figure creation and styling.
    
    Usage:
        FIG = FigFinalizer()
        
        @FIG()
        def plot_something(ax, data, palette=None):
            ax.plot(data["x"], data["y"])
            return {"handles": None, "labels": None}  # optional

        # In caller:
        FIG.show(plot_something, data=df, title="My Title", xlabel="X", ylabel="Y")
    
    Alternatively, use it as a context when composing multi-panel layouts:
        fig, axes = FIG.new_figure(ncols=2, sharey=True)
        plot_left(axes[0], ...)
        plot_right(axes[1], ...)
        FIG.finalize(fig, axes, suptitle="Combo figure")
    """
    def __init__(self, theme: Optional[PlotTheme] = None, default_save_dir: Optional[str] = None, show_default: bool = True):
        self.theme = theme or PlotTheme()
        self.default_save_dir = default_save_dir
        self.show_default = show_default

    # ---------- low-level helpers ----------

    def new_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[float, float]] = None,
        sharex: bool = False,
        sharey: bool = False,
        squeeze: bool = True,
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        fig = plt.figure(figsize=figsize or self.theme.figsize, dpi=self.theme.dpi, constrained_layout=self.theme.constrained_layout)
        axes = fig.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, squeeze=squeeze)
        return fig, axes

    def _apply_axes_style(
        self,
        ax: plt.Axes,
        *,
        title: Optional[str],
        xlabel: Optional[str],
        ylabel: Optional[str],
        legend: Union[bool, str],
        legend_loc: str,
        legend_title: Optional[str],
        group_labels: Optional[Sequence[str]],
    ):
        if title is not None:
            ax.set_title(title, fontsize=self.theme.title_size)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=self.theme.label_size)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=self.theme.label_size)

        # Grid
        if self.theme.grid:
            ax.grid(True, linestyle=self.theme.grid_style, alpha=self.theme.grid_alpha)

        # Zero reference
        if self.theme.zero_line:
            ax.axhline(0.0, color="0.25", linewidth=1, linestyle="--", alpha=0.6, zorder=0)

        # Ticks
        for tick in ax.get_xticklabels():
            tick.set_fontsize(self.theme.tick_size)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(self.theme.tick_size)

        # Legend
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            # Optionally override labels if provided via theme/kwargs
            if group_labels:
                if len(labels) == 0 or len(labels) == len(group_labels):
                    labels = list(group_labels)
            if len(labels) > 0:
                ax.legend(
                    handles,
                    labels,
                    loc=legend_loc,
                    fontsize=self.theme.legend_size,
                    frameon=False,
                    title=legend_title,
                )

    def finalize(
        self,
        fig: plt.Figure,
        axes: Union[plt.Axes, Iterable[plt.Axes]],
        *,
        suptitle: Optional[str] = None,
        save: Optional[str] = None,
        show: Optional[bool] = None,
        tight_layout: Optional[bool] = None,
    ) -> plt.Figure:
        if isinstance(axes, np.ndarray):
            ax_list = axes.ravel().tolist()
        elif isinstance(axes, (list, tuple)):
            ax_list = list(axes)
        else:
            ax_list = [axes]

        # Layout
        if tight_layout if tight_layout is not None else self.theme.tight_layout:
            fig.tight_layout()

        # Suptitle after tight_layout (or constrained_layout handles it)
        if suptitle:
            fig.suptitle(suptitle, fontsize=self.theme.title_size, y=1.02)

        # Save
        if save:
            path = save
            if self.default_save_dir and not os.path.isabs(save):
                os.makedirs(self.default_save_dir, exist_ok=True)
                path = os.path.join(self.default_save_dir, save)
            fig.savefig(path, dpi=self.theme.dpi, bbox_inches="tight")

        # Show
        if show if show is not None else self.show_default:
            plt.show()

        return fig

    def __call__(self, **preset_style):
        """Return a decorator that wraps a plotting function.
        
        The wrapped function should accept an `ax` kwarg and draw artists.
        It should NOT set titles/labels/legend-those are handled here.
        """
        def decorator(plot_func: Callable[..., Dict[str, Any]]):
            def wrapper(
                *args,
                title: Optional[str] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                legend: Union[bool, str] = "auto",
                legend_loc: str = "best",
                save: Optional[str] = None,
                show: Optional[bool] = None,
                ax: Optional[plt.Axes] = None,
                figsize: Optional[Tuple[float, float]] = None,
                palette: Optional[Sequence[str]] = None,
                **kwargs,
            ) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
                # 1) Ensure Axes
                created = False
                if ax is None:
                    fig, ax = self.new_figure(figsize=figsize)[0:2]
                    created = True
                else:
                    fig = ax.get_figure()

                # 2) Merge preset style with call-site overrides
                style = dict(preset_style)
                style.update(
                    dict(
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        legend=legend,
                        legend_loc=legend_loc,
                        legend_title=kwargs.pop("legend_title", self.theme.legend_title),
                        group_labels=kwargs.pop("group_labels", self.theme.group_labels),
                    )
                )

                # 3) Call the plot function
                out = plot_func(*args, ax=ax, palette=(palette or self.theme.palette), **kwargs) or {}

                # 4) Style
                self._apply_axes_style(ax, **style)

                # 5) Finish
                if created:
                    self.finalize(fig, ax, save=save, show=show)

                return fig, ax, out
            return wrapper
        return decorator

    # Convenience: one-shot call for simple single-axes plots
    def show(self, plot_func: Callable, *args, **kwargs):
        decorated = self(**{})(plot_func)
        return decorated(*args, **kwargs)


# Global instance used by plotting helpers below
FIG = FigFinalizer()


# ================================
# Data drawing functions
# ================================

@FIG(title=None, xlabel=None, ylabel=None)
def plot_dose_distribution(
    dose: Union[pd.Series, np.ndarray],
    ax: plt.Axes,
    palette: Sequence[str],
    bin_edges: Optional[Sequence[float]] = None,
    bin_labels: Optional[Sequence[str]] = None,
    bins: int = 40,
    density: bool = False,
) -> Dict[str, Any]:
    """
    Histogram of dose levels with optional vertical lines at bin edges.

    If bin_edges+bin_labels are provided, each bar corresponds to a dose bin
    and is colored using the global Small/Medium/Large color policy.
    """
    x = pd.Series(dose).dropna().values
    if x.size == 0:
        return {"n": 0}

    cmap = _dose_color_map_from_palette(palette)

    if bin_edges is not None:
        # Replace inf with finite upper bound for plotting
        edges_in = []
        x_max = float(np.nanmax(x))
        for e in bin_edges:
            edges_in.append(float(e) if np.isfinite(e) else x_max * 1.01)
        edges_in = sorted(edges_in)
        if len(edges_in) < 2:
            edges_in = np.linspace(np.nanmin(x), np.nanmax(x), bins if bins > 1 else 2).tolist()

        # Drop below smallest edge (so zeros don't dominate)
        lower = min(edges_in)
        x = x[x >= lower]
        if x.size == 0:
            return {"n": 0, "counts": [], "edges": edges_in}

        # vertical lines at edges
        for e in edges_in:
            ax.axvline(e, linestyle="--", linewidth=1, color="0.3", alpha=0.7, zorder=0)

        counts, edges_out, patches = ax.hist(x, bins=edges_in, density=density, alpha=0.9)

        # Color bars by bin label (Small/Medium/Large)
        for i, patch in enumerate(patches):
            lbl = None
            if bin_labels is not None and i < len(bin_labels):
                lbl = str(bin_labels[i])
                patch.set_label(lbl)

            canon = _canonical_bin_label(lbl) if lbl is not None else None
            color = cmap.get(canon, palette[(i % len(palette))])
            patch.set_facecolor(color)
            patch.set_edgecolor("white")

    else:
        counts, edges_out, patches = ax.hist(
            x, bins=bins, density=density, color=palette[0], alpha=0.85
        )

    return {"n": len(x), "counts": counts, "edges": edges_out.tolist()}






@FIG(title="ATT$^{o}$ (pooled)", xlabel=None, ylabel="Effect (Deltalog units)")
def plot_att_pooled_point(
    coef: float,
    lo: float,
    hi: float,
    ax: plt.Axes,
    palette: Sequence[str],
    marker: str = "o",
) -> Dict[str, Any]:
    """Single-point estimate with CI."""
    ax.errorbar([0], [coef], yerr=[[coef - lo], [hi - coef]], fmt=marker, color=palette[0], capsize=4, linewidth=2)
    ax.set_xlim(-1, 1)
    return {}

@FIG(
    title="Event study (Callaway–Sant'Anna aggregation)",
    xlabel=r"Event time ($\tau = t - g$)",
    ylabel="Effect (Deltalog units)",
)
def plot_differences_event_agg(
    df_event: pd.DataFrame,
    ax: plt.Axes,
    palette: Sequence[str],
    *,
    ref_tau: int = -1,
    max_pre: int | None = 5,     # show at most 10 years before treatment
    max_post: int | None = 10,    # and 10 years after; tweak as needed
    xtick_step: int = 2,          # only label every 2nd tick
) -> Dict[str, Any]:
    """
    Plot event-study aggregation from
        differences.ATTgt.aggregate(type_of_aggregation="event").

    Follows standard CS / did practice:
      - x-axis: event time (tau = t - g)
      - y-axis: ATT(τ)
      - red dots for pre-treatment (tau < 0)
      - blue dots for post-treatment (tau >= 0)
      - 95% simultaneous confidence band as vertical error bars
      - vertical line at ref_tau (usually -1 = last pre-period)

    The plotting range is cropped to [ -max_pre, max_post ] to improve
    readability; estimation still uses the full range.
    """

    # 1) Tidy and sort
    d = tidy_differences_event_agg_df(df_event).copy()
    if d.empty:
        return {"n": 0}

    d = d.sort_values("event_time")
    d["event_time"] = d["event_time"].astype(float)
    d["beta"] = d["beta"].astype(float)
    d["lo"] = d["lo"].astype(float)
    d["hi"] = d["hi"].astype(float)

    # 2) Optional cropping of the event-time window for plotting
    d["is_pre"] = d["event_time"] < 0

    if max_pre is not None:
        d = d[~d["is_pre"] | (d["event_time"] >= -float(max_pre))]

    if max_post is not None:
        d = d[d["event_time"] <= float(max_post)]

    if d.empty:
        return {"n": 0}

    tau = d["event_time"].to_numpy()
    beta = d["beta"].to_numpy()
    lo = d["lo"].to_numpy()
    hi = d["hi"].to_numpy()

    if not np.isfinite(tau).any() or not np.isfinite(beta).any():
        return {"n": 0}

    # 3) Split pre vs post
    ref_val = float(ref_tau) if ref_tau is not None else -1.0
    # Drop the reference period (tau ~= ref_val, typically -1) so it does not appear as its own point
    ref_mask = np.isclose(tau, ref_val)
    if ref_mask.any():
        keep = ~ref_mask
        tau = tau[keep]
        beta = beta[keep]
        lo = lo[keep]
        hi = hi[keep]

    pre_mask = tau < ref_val            # strictly before ref period (e.g., tau <= -2)
    post_mask = tau >= 0                # start post at 0 and above

    color_post = palette[0]                    # blue-ish
    color_pre = palette[3] if len(palette) > 3 else "0.4"  # red-ish / grey

    # horizontal zero line for reference
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)

    # 4) Draw pre- and post-period series with error bars + lines
    for mask, color, label in [
        (pre_mask, color_pre, "Pre-treatment"),
        (post_mask, color_post, "Post-treatment"),
    ]:
        if not mask.any():
            continue
        x = tau[mask]
        y = beta[mask]
        ylo = lo[mask]
        yhi = hi[mask]

        ax.errorbar(
            x,
            y,
            yerr=[y - ylo, yhi - y],
            fmt="o",
            color=color,
            capsize=3,
            linewidth=1.3,
            markersize=5,
            label=label,
        )
        ax.plot(x, y, color=color, linewidth=1.3, alpha=0.7)

    # Lines to ensure continuity at the ref period
    if pre_mask.any():
        x_pre = tau[pre_mask]
        y_pre = beta[pre_mask]
        ax.plot(x_pre, y_pre, color=color_pre, linewidth=1.3, alpha=0.9)

    if post_mask.any():
        x_post = tau[post_mask]
        y_post = beta[post_mask]
        if pre_mask.any():
            x_post = np.insert(x_post, 0, x_pre[-1])
            y_post = np.insert(y_post, 0, y_pre[-1])
        ax.plot(x_post, y_post, color=color_post, linewidth=1.3, alpha=0.9)

    # 5) Integer x-ticks over observed tau range, thinned by xtick_step
    tmin = int(np.floor(np.nanmin(tau)))
    tmax = int(np.ceil(np.nanmax(tau)))
    ticks = np.arange(tmin, tmax + 1, xtick_step)
    ax.set_xticks(ticks)

    # 6) Reference last pre-period (usually tau = -1)
    if ref_tau is not None:
        ax.axvline(float(ref_tau), color="0.2", linestyle="-", linewidth=1.0, alpha=0.7)

    # Optional: move legend out of the way
    ax.legend(loc="upper left", frameon=False)

    return {
        "n": int(len(d)),
        "tmin": tmin,
        "tmax": tmax,
    }




@FIG(title="ATT$^{o}$ (pooled + by bin)", xlabel=None, ylabel="Effect (Deltalog units)")
def plot_att_combined(
    coef: float,
    lo: float,
    hi: float,
    bins_df: pd.DataFrame,
    ax: plt.Axes,
    palette: Sequence[str],
    marker: str = "o",
    rotate_labels: int = 20,
) -> Dict[str, Any]:
    """
    ATT^o pooled + binned shown as points with 95% CI error bars
    (instead of the current box-plot representation).
    """
    # 1) Build plotting rows: pooled first, then bins in their given order
    rows = []
    rows.append({"label": "Pooled", "coef": float(coef), "lo": float(lo), "hi": float(hi)})
    # Expecting columns ['bin','coef','lo','hi'] in bins_df
    for _, r in bins_df.iterrows():
        rows.append({"label": str(r["bin"]), "coef": float(r["coef"]), "lo": float(r["lo"]), "hi": float(r["hi"])})

    # 2) X positions and aesthetics
    x = np.arange(len(rows))
    labels = [r["label"] for r in rows]

    # 3) Draw: vertical errorbars with centered point
    for i, r in enumerate(rows):
        y = r["coef"]
        yerr = np.array([[y - r["lo"]], [r["hi"] - y]])  # asymmetric (lo, hi)
        ax.errorbar(
            x[i], y, yerr=yerr, fmt=marker, capsize=3, elinewidth=1.5, linewidth=0,
            zorder=3, color=palette[0] if i == 0 else palette[(i % (len(palette) - 1)) + 1]
        )

    # 4) Reference line and axes cosmetics
    ax.axhline(0, linestyle="--", linewidth=1, color="0.6", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate_labels, ha="right")
    ax.set_xlim(-0.5, len(x) - 0.5)

    return {}


@FIG(title="ATT$^{o}$ by dose bin", xlabel=None, ylabel="Effect (Deltalog units)")
def plot_att_binned_points(
    bins_labels: Sequence[str],
    coef: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    ax: plt.Axes,
    palette: Sequence[str],
    rotate_labels: int = 20,
) -> Dict[str, Any]:
    """Per-bin point estimates with CIs (fixed bug: uses hi[i]-y[i])."""
    x = np.arange(len(bins_labels))
    y = np.asarray(coef, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    for i in range(len(x)):
        ax.errorbar(
            [x[i]], [y[i]],
            yerr=[[y[i] - lo[i]], [hi[i] - y[i]]],
            fmt="o", color=palette[i % len(palette)], capsize=4, linewidth=2
        )
    ax.set_xticks(x)
    ax.set_xticklabels(bins_labels, rotation=rotate_labels, ha="right")
    ax.set_xlim(-0.5, len(x) - 0.5)
    return {}


@FIG(title="Event study (pooled)", xlabel="Event time tau", ylabel="Effect (Deltalog units)")
def plot_event_study_line(
    df_es: pd.DataFrame,
    ax: plt.Axes,
    palette: Sequence[str],
    alpha_band: float = 0.15,
    ref_tau: int = -1,
) -> Dict[str, Any]:
    """Point estimates with error bars (95% CI) for event-study."""
    # Expect columns: event_time, beta, lo, hi (or se)
    d = df_es.copy().sort_values("event_time")
    beta = d["beta"].astype(float).values
    x = d["event_time"].astype(float).values

    if "lo" in d and "hi" in d:
        lo = d["lo"].astype(float).values
        hi = d["hi"].astype(float).values
    elif "se" in d:
        # 95% normal approx
        lo = beta - 1.96 * d["se"].astype(float).values
        hi = beta + 1.96 * d["se"].astype(float).values
    else:
        raise ValueError("df_es must contain either (lo, hi) or se.")

    # Colour scheme: pre (<= ref_tau) red, post (> ref_tau) blue
    ref_val = float(ref_tau) if ref_tau is not None else -1.0
    pre_mask = x <= ref_val
    post_mask = x > ref_val
    color_post = palette[0]  # blue
    color_pre = palette[3] if len(palette) > 3 else "0.4"  # red-ish

    # Plot error bars and markers
    for mask, color, label in [
        (pre_mask, color_pre, "Pre-treatment"),
        (post_mask, color_post, "Post-treatment"),
    ]:
        if not mask.any():
            continue
        xm = x[mask]
        ym = beta[mask]
        ylo = lo[mask]
        yhi = hi[mask]
        ax.errorbar(
            xm,
            ym,
            yerr=[ym - ylo, yhi - ym],
            fmt="o",
            color=color,
            capsize=4,
            linewidth=2,
            markersize=6,
            label=label,
        )

    # Lines to connect segments, ensuring post starts from the ref period
    if pre_mask.any():
        x_pre = x[pre_mask]
        y_pre = beta[pre_mask]
        ax.plot(x_pre, y_pre, linewidth=1.6, color=color_pre, alpha=0.85, zorder=0)

    if post_mask.any():
        x_post = x[post_mask]
        y_post = beta[post_mask]
        if pre_mask.any():
            # prepend last pre point so the post line connects to ref period
            x_post = np.insert(x_post, 0, x_pre[-1])
            y_post = np.insert(y_post, 0, y_pre[-1])
        ax.plot(x_post, y_post, linewidth=1.6, color=color_post, alpha=0.85, zorder=0)

    # Reference tau = -1
    if ref_tau is not None:
        ax.axvline(float(ref_tau), color="0.2", linestyle="-", linewidth=1.2, alpha=0.7)

    return {}


@FIG(title="HonestDiD M-sensitivity", xlabel="M (relative magnitude bound)", ylabel="Bounded effect interval")
def plot_honest_did_M_curve(
    m_grid: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    theta_hat: Optional[float],
    ax: plt.Axes,
    palette: Sequence[str],
    show_frontier: bool = True,
) -> Dict[str, Any]:
    """Plot HonestDiD lower/upper bounds vs M and overlay theta_hat."""
    m = np.asarray(m_grid, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    ax.plot(m, lo, linewidth=2, color=palette[1], label="Lower bound")
    ax.plot(m, hi, linewidth=2, color=palette[2], label="Upper bound")
    ax.fill_between(m, lo, hi, alpha=0.10, color=palette[2])

    if theta_hat is not None and np.isfinite(theta_hat):
        ax.axhline(theta_hat, color=palette[0], linewidth=2, linestyle=":", label=r"$\hat{\theta}$")

    # Frontier M*: smallest M where 0 first enters [lo, hi]
    if show_frontier:
        inside = (lo <= 0) & (0 <= hi)
        if inside.any():
            idx = int(np.argmax(inside))  # first True
            ax.axvline(m[idx], color="0.4", linestyle="--", label=f"M*~={m[idx]:.2g}")

    return {}


@FIG(
    title="Callaway–Sant'Anna ATT(g,t) by cohort",
    xlabel=r"Event time ($\\tau = t - g$)",
    ylabel="ATT(g, t)",
)
def plot_differences_att_gt(
    att_res: DifferencesAttResult,
    ax: plt.Axes,
    palette: Sequence[str],
    *,
    alpha_band: float = 0.12,
    min_points: int = 2,
) -> Dict[str, Any]:
    """Plot cohort × time ATTgt estimates from DifferencesAttResult.

    Expects att_res.att_gt_df with columns: 'g', 't', 'att' (and optional 'se').
    Groups by cohort g and draws series against event time tau = t - g.
    """
    if att_res is None or getattr(att_res, "att_gt_df", None) is None:
        return {"n_groups": 0, "rows": 0}

    df = att_res.att_gt_df.copy()
    if df is None or len(df) == 0 or ("att" in df and df["att"].isna().all()):
        return {"n_groups": 0, "rows": 0}

    # Ensure required columns
    for req in ("g", "t", "att"):
        if req not in df.columns:
            return {"n_groups": 0, "rows": 0}

    # Event time
    df["tau"] = df["t"].astype(float) - df["g"].astype(float)
    df = df.dropna(subset=["tau", "att"]).copy()
    if df.empty:
        return {"n_groups": 0, "rows": 0}

    # Draw reference lines
    ax.axvline(x=-1, color="0.3", linestyle=":", linewidth=1.0, alpha=0.8, zorder=0)

    # Plot per cohort
    color_post = palette[0]
    color_pre = palette[3] if len(palette) > 3 else "0.4"
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    linestyles = ["-", "--", "-.", ":"]
    n_groups = 0
    for idx, (g, gdf) in enumerate(df.groupby("g", dropna=True)):
        gdf = gdf.sort_values("tau")
        if len(gdf) < min_points:
            continue
        label = f"Cohort g={int(g)}" if float(g).is_integer() else f"Cohort g={g}"
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]

        x = gdf["tau"].astype(float).values
        y = gdf["att"].astype(float).values
        se = gdf["se"].astype(float).values if "se" in gdf.columns else None
        label_used = False

        # Colour pre-period red and post-period blue across plots
        for mask, color in [(x < 0, color_pre), (x >= 0, color_post)]:
            if not mask.any():
                continue
            xm = x[mask]
            ym = y[mask]
            ax.plot(
                xm,
                ym,
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=label if not label_used else None,
                linewidth=1.6,
                alpha=0.85,
                zorder=2,
            )

            if se is not None and np.isfinite(se[mask]).any():
                se_masked = se[mask]
                lo = ym - 1.96 * se_masked
                hi = ym + 1.96 * se_masked
                ax.fill_between(xm, lo, hi, color=color, alpha=alpha_band, zorder=1)
            label_used = True

        n_groups += 1

    # Integer x-ticks over observed tau range
    if df["tau"].notna().any():
        tmin = int(np.floor(df["tau"].min()))
        tmax = int(np.ceil(df["tau"].max()))
        ax.set_xticks(np.arange(tmin, tmax + 1))

    return {"n_groups": n_groups, "rows": int(len(df))}


@FIG(title="Pre-trend support (units by lead tau)", xlabel="Event time tau (leads only)", ylabel="Units")
def plot_support_by_tau(
    df_support: pd.DataFrame,
    ax: plt.Axes,
    palette: Sequence[str],
    pre_window: int = 10,
) -> Dict[str, Any]:
    """Bar chart: number of units contributing at each lead tau (pooled)."""
    # Expect columns: event_time, units (>=0)
    d = df_support.copy()
    d = d[(d["event_time"] < 0) & (d["event_time"] >= -pre_window)]
    d = d.sort_values("event_time")
    ax.bar(d["event_time"], d["units"], color=palette[0])
    return {}


from matplotlib import colors as mcolors

def _clip_nonnegative_ci(mu: np.ndarray, lo: np.ndarray, hi: np.ndarray, floor: float = 0.0):
    """
    Ensure nonnegative plotting for a physically nonnegative quantity.
    Clips lo/hi (and mu if needed) at `floor`, and enforces lo <= mu <= hi.
    """
    mu2 = np.maximum(mu, floor)
    lo2 = np.maximum(lo, floor)
    hi2 = np.maximum(hi, floor)

    # enforce ordering for errorbar yerr safety
    lo2 = np.minimum(lo2, mu2)
    hi2 = np.maximum(hi2, mu2)
    return mu2, lo2, hi2

@FIG(title="Mean dose by event time", xlabel="Event time tau", ylabel="Mean dose")
def plot_mean_dose_by_tau(
    df_panel: pd.DataFrame,
    ax: plt.Axes,
    palette: Sequence[str],
    dose_col: str = "dose_level",
    event_time_col: str = "event_time",
    bin_col: Optional[str] = "dose_bin",
    ref_tau: int = -1,
    show_pooled: bool = True,
) -> Dict[str, Any]:
    """
    Consistent colors:
      - Pooled: palette[0]
      - Small/Medium/Large: palette[1]/palette[2]/palette[4]
    """
    cmap = _dose_color_map_from_palette(palette)

    needed_cols = [dose_col, event_time_col]
    if bin_col and bin_col in df_panel.columns:
        needed_cols.append(bin_col)

    df = df_panel[needed_cols].copy()
    df = df.dropna(subset=[dose_col, event_time_col])
    if df.empty:
        return {"n": 0, "n_series": 0}

    has_bins = bool(bin_col and bin_col in df.columns and df[bin_col].notna().any())

    # --- pooled
    if show_pooled:
        gp = (
            df.groupby(event_time_col)[dose_col]
              .agg(["mean", "std", "count"])
              .reset_index()
              .sort_values(event_time_col)
        )
        gp["std"] = gp["std"].fillna(0.0)

        # SEM defined only when count>=2; otherwise no CI (lo=hi=mean)
        gp["sem"] = np.where(gp["count"] >= 2, gp["std"] / np.sqrt(gp["count"]), np.nan)

        gp["ci_low"]  = np.where(gp["count"] >= 2, gp["mean"] - 1.96 * gp["sem"], gp["mean"])
        gp["ci_high"] = np.where(gp["count"] >= 2, gp["mean"] + 1.96 * gp["sem"], gp["mean"])


        tau = gp[event_time_col].astype(float).values
        mu  = gp["mean"].astype(float).values
        lo  = gp["ci_low"].astype(float).values
        hi  = gp["ci_high"].astype(float).values

        # --- CLIP HERE (dose/capacity is nonnegative)
        mu, lo, hi = _clip_nonnegative_ci(mu, lo, hi, floor=0.0)


        c = cmap["Pooled"]
        for i in range(len(tau)):
            ax.errorbar(
                [tau[i]], [mu[i]],
                yerr=[[mu[i] - lo[i]], [hi[i] - mu[i]]],
                fmt="o", color=c, capsize=4, linewidth=2, markersize=6, alpha=0.6
            )
        ax.plot(tau, mu, linewidth=2, color=c, alpha=0.6, zorder=0, label="Pooled")

    # --- by-bin
    ordered_bins_present: list[str] = []
    if has_bins:
        df["_dose_bin_label"] = df[bin_col].map(_canonical_bin_label)

        gb = (
            df.groupby([event_time_col, "_dose_bin_label"])[dose_col]
              .agg(["mean", "std", "count"])
              .reset_index()
              .sort_values([event_time_col, "_dose_bin_label"])
        )
        gb["std"] = gb["std"].fillna(0.0)
        gb["sem"] = np.where(gb["count"] >= 2, gb["std"] / np.sqrt(gb["count"]), np.nan)
        gb["ci_low"]  = np.where(gb["count"] >= 2, gb["mean"] - 1.96 * gb["sem"], gb["mean"])
        gb["ci_high"] = np.where(gb["count"] >= 2, gb["mean"] + 1.96 * gb["sem"], gb["mean"])


        present = set(gb["_dose_bin_label"].dropna().astype(str).unique())
        ordered_bins_present = [b for b in DOSE_BIN_ORDER if b in present] + [b for b in sorted(present) if b not in set(DOSE_BIN_ORDER)]

        for b in ordered_bins_present:
            bd = gb[gb["_dose_bin_label"].astype(str) == b].sort_values(event_time_col)
            if bd.empty:
                continue

            tau = bd[event_time_col].astype(float).values
            mu  = bd["mean"].astype(float).values
            lo  = bd["ci_low"].astype(float).values
            hi  = bd["ci_high"].astype(float).values

            # --- CLIP HERE too
            mu, lo, hi = _clip_nonnegative_ci(mu, lo, hi, floor=0.0)


            c = cmap.get(b, palette[1])
            for i in range(len(tau)):
                ax.errorbar(
                    [tau[i]], [mu[i]],
                    yerr=[[mu[i] - lo[i]], [hi[i] - mu[i]]],
                    fmt="o", color=c, capsize=4, linewidth=2, markersize=5, alpha=0.6
                )
            ax.plot(tau, mu, linewidth=1.5, color=c, alpha=0.6, zorder=0, label=b)

        ax.legend(loc="best", fontsize=10, frameon=False)

    if ref_tau is not None:
        ax.axvline(float(ref_tau), color="0.2", linestyle="-", linewidth=1.2, alpha=0.7)
    #ax.set_ylim(bottom=0.0)

    n_series = (1 if show_pooled else 0) + (len(ordered_bins_present) if ordered_bins_present else 0)
    return {"n": len(df), "n_series": n_series}



# ================================
# Composite helpers (optional)
# ================================

def plot_att_combo(
    pooled: Dict[str, float],
    binned_df: pd.DataFrame,
    es_df: pd.DataFrame,
    *, suptitle: Optional[str] = "ATT$^{o}$ summary",
    save: Optional[str] = None,
    show: Optional[bool] = None,
) -> plt.Figure:
    """Convenience: pooled point, binned points, and ES in one row."""
    fig, axes = FIG.new_figure(ncols=3, figsize=(16, 5))
    # Pooled
    plot_att_pooled_point(
        coef=float(pooled["coef"]),
        lo=float(pooled["lo"]),
        hi=float(pooled["hi"]),
        ax=axes[0],
        title="ATT$^{o}$ (pooled)",
        ylabel="Effect (Deltalog units)",
    )
    # Binned
    plot_att_binned_points(
        bins_labels=list(binned_df["bin"]),
        coef=binned_df["coef"].astype(float).values,
        lo=binned_df["lo"].astype(float).values,
        hi=binned_df["hi"].astype(float).values,
        ax=axes[1],
        title="ATT$^{o}$ by dose bin",
        ylabel=None,
    )
    # ES
    plot_event_study_line(
        es_df, ax=axes[2], title="Event study (pooled)", xlabel="Event time tau", ylabel=None
    )
    FIG.finalize(fig, axes, suptitle=suptitle, save=save, show=show)
    return fig
