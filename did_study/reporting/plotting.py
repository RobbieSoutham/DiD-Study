
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union, Dict, Any

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================
# Theme + Figure Finalizer
# ================================

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

    def _apply_axes_style(self, ax: plt.Axes, *, title: Optional[str], xlabel: Optional[str], ylabel: Optional[str], legend: Union[bool, str], legend_loc: str):
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
            if len(labels) > 0:
                ax.legend(handles, labels, loc=legend_loc, fontsize=self.theme.legend_size, frameon=False)

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
        It should NOT set titles/labels/legend—those are handled here.
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
                style.update(dict(title=title, xlabel=xlabel, ylabel=ylabel, legend=legend, legend_loc=legend_loc))

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
    bins: int = 40,
    density: bool = False,
) -> Dict[str, Any]:
    """Histogram of dose levels with optional vertical lines at bin edges."""
    x = pd.Series(dose).dropna().values
    if bin_edges is not None:
        # if edges include inf, drop for plotting vertical lines
        finite_edges = [e for e in bin_edges if np.isfinite(e)]
        for e in finite_edges:
            ax.axvline(float(e), linestyle="--", linewidth=1, color="0.3", alpha=0.7, zorder=0)
        # For histogram bins, clip to max observed
        edges = [b for b in bin_edges if np.isfinite(b)]
        if len(edges) == 0:
            edges = np.linspace(np.nanmin(x), np.nanmax(x), 20).tolist()
        ax.hist(x, bins=edges + [float(np.nanmax(x))], density=density, color=palette[0], alpha=0.85)
    else:
        ax.hist(x, bins=bins, density=density, color=palette[0], alpha=0.85)
    return {"n": len(x)}


@FIG(title="ATT$^{o}$ (pooled)", xlabel=None, ylabel="Effect (Δlog units)")
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


@FIG(title="ATT$^{o}$ (pooled + by bin)", xlabel=None, ylabel="Effect (Δlog units)")
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
    """Box plot: pooled estimate and binned estimates with confidence intervals.
    
    The box represents the 95% confidence interval (q1 to q3), with the median line
    at the point estimate. Since we're showing confidence intervals (not data distributions),
    the whiskers are set equal to the CI bounds (no tails extending beyond). This is
    appropriate for CI visualization where the box itself represents the uncertainty range.
    """
    # Prepare data for box plot
    positions = []
    labels = []
    data_for_box = []
    
    # Add pooled estimate
    positions.append(0)
    labels.append("Pooled")
    # Box spans the 95% CI (q1=lo to q3=hi), whiskers at same bounds (no tails)
    # This correctly represents that the CI is the uncertainty range
    pooled_data = {
        'med': coef,
        'q1': lo,
        'q3': hi,
        'whislo': lo,  # Whisker at lower CI bound (no tail beyond)
        'whishi': hi,  # Whisker at upper CI bound (no tail beyond)
        'mean': coef,  # Required when showmeans=True
    }
    data_for_box.append(pooled_data)
    
    # Add binned estimates
    if len(bins_df) > 0:
        bins_labels = list(bins_df["bin"].astype(str))
        coefs = bins_df["coef"].astype(float).values
        los = bins_df["lo"].astype(float).values
        his = bins_df["hi"].astype(float).values
        
        for i in range(len(bins_df)):
            positions.append(i + 1)
            labels.append(bins_labels[i])
            bin_data = {
                'med': coefs[i],
                'q1': los[i],
                'q3': his[i],
                'whislo': los[i],  # Whisker at lower CI bound
                'whishi': his[i],  # Whisker at upper CI bound
                'mean': coefs[i],  # Required when showmeans=True
            }
            data_for_box.append(bin_data)
    
    # Create box plot using bxp
    bp = ax.bxp(
        data_for_box,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=False,  # Don't show mean marker since we have median
        meanline=False,
        showfliers=False,
        showcaps=True,  # Show caps on whiskers (will appear at CI bounds)
        manage_ticks=False,  # We'll set ticks manually
    )
    
    # Color the boxes
    for i, patch in enumerate(bp['boxes']):
        if i == 0:
            patch.set_facecolor(palette[0])
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor(palette[(i - 1) % len(palette) + 1])
            patch.set_alpha(0.7)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(0, color='0.5', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    
    # Set x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotate_labels, ha="right")
    ax.set_xlim(-0.5, len(positions) - 0.5)
    
    return {}


@FIG(title="ATT$^{o}$ by dose bin", xlabel=None, ylabel="Effect (Δlog units)")
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


@FIG(title="Event study (pooled)", xlabel="Event time τ", ylabel="Effect (Δlog units)")
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

    # Plot with error bars (like ATT^o by dose bin)
    for i in range(len(x)):
        ax.errorbar(
            [x[i]], [beta[i]],
            yerr=[[beta[i] - lo[i]], [hi[i] - beta[i]]],
            fmt="o", color=palette[0], capsize=4, linewidth=2, markersize=6
        )
    
    # Connect points with a line
    ax.plot(x, beta, linewidth=1.5, color=palette[0], alpha=0.5, zorder=0)

    # Reference τ = -1
    if ref_tau is not None:
        ax.axvline(float(ref_tau), color="0.2", linestyle="-", linewidth=1.2, alpha=0.7)

    return {}


@FIG(title="HonestDiD M-sensitivity", xlabel="M (relative magnitude bound)", ylabel="Bounded effect interval")
def plot_honestdid_mcurve(
    m_grid: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    theta_hat: Optional[float],
    ax: plt.Axes,
    palette: Sequence[str],
    show_frontier: bool = True,
) -> Dict[str, Any]:
    """Plot HonestDiD lower/upper bounds vs M and overlay θ̂."""
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
            ax.axvline(m[idx], color="0.4", linestyle="--", label=f"M*≈{m[idx]:.2g}")

    return {}


@FIG(title="Pre-trend support (units by lead τ)", xlabel="Event time τ (leads only)", ylabel="Units")
def plot_support_by_tau(
    df_support: pd.DataFrame,
    ax: plt.Axes,
    palette: Sequence[str],
) -> Dict[str, Any]:
    """Bar chart: number of units contributing at each lead τ (pooled)."""
    # Expect columns: event_time, units (>=0)
    d = df_support.copy()
    d = d[d["event_time"] < 0].sort_values("event_time")
    ax.bar(d["event_time"], d["units"], color=palette[0])
    return {}


@FIG(title="Mean dose by event time", xlabel="Event time τ", ylabel="Mean dose")
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
    """Plot mean dose by event time with confidence intervals, showing both pooled and by bin.
    
    This diagnostic plot helps assess whether treatment intensity varies
    systematically with event time, which could confound the event study.
    Shows pooled aggregate AND separate series for each bin if bins are available.
    """
    # Filter to rows with valid dose and event_time
    needed_cols = [dose_col, event_time_col]
    if bin_col and bin_col in df_panel.columns:
        needed_cols.append(bin_col)
    
    df = df_panel[needed_cols].copy()
    df = df.dropna(subset=[dose_col, event_time_col])
    
    if len(df) == 0:
        return {"n": 0}
    
    # Check if we have bins
    has_bins = bin_col and bin_col in df.columns and df[bin_col].notna().any()
    
    # Always plot pooled aggregate first
    if show_pooled:
        grouped_pooled = df.groupby(event_time_col)[dose_col].agg(['mean', 'std', 'count']).reset_index()
        grouped_pooled = grouped_pooled.sort_values(event_time_col)
        # Use SEM for CI, but handle cases with few observations
        # For n < 2, std is NaN, so we set it to 0
        grouped_pooled['std'] = grouped_pooled['std'].fillna(0.0)
        grouped_pooled['sem'] = grouped_pooled['std'] / np.sqrt(grouped_pooled['count'].clip(lower=2))
        # Only show CI if we have at least 2 observations
        grouped_pooled['ci_low'] = np.where(
            grouped_pooled['count'] >= 2,
            grouped_pooled['mean'] - 1.96 * grouped_pooled['sem'],
            grouped_pooled['mean']
        )
        grouped_pooled['ci_high'] = np.where(
            grouped_pooled['count'] >= 2,
            grouped_pooled['mean'] + 1.96 * grouped_pooled['sem'],
            grouped_pooled['mean']
        )
        
        tau_pooled = grouped_pooled[event_time_col].astype(float).values
        mean_pooled = grouped_pooled["mean"].astype(float).values
        ci_low_pooled = grouped_pooled["ci_low"].astype(float).values
        ci_high_pooled = grouped_pooled["ci_high"].astype(float).values
        
        # Plot pooled with error bars
        for i in range(len(tau_pooled)):
            ax.errorbar(
                [tau_pooled[i]], [mean_pooled[i]],
                yerr=[[mean_pooled[i] - ci_low_pooled[i]], [ci_high_pooled[i] - mean_pooled[i]]],
                fmt="o", color=palette[0], capsize=4, linewidth=2, markersize=6
            )
        
        # Connect pooled points with a line
        ax.plot(tau_pooled, mean_pooled, linewidth=2, color=palette[0], alpha=0.7, zorder=0, label="Pooled")
    
    # Then plot bins if available
    if has_bins:
        # Group by event_time and bin, compute mean and CI
        grouped = df.groupby([event_time_col, bin_col])[dose_col].agg(['mean', 'std', 'count']).reset_index()
        grouped = grouped.sort_values([event_time_col, bin_col])
        
        # Compute 95% CI using SEM (standard error of mean)
        # SEM = std / sqrt(n), CI = mean ± 1.96 * SEM
        # Handle cases with few observations
        grouped['std'] = grouped['std'].fillna(0.0)
        grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'].clip(lower=2))
        # Only show CI if we have at least 2 observations
        grouped['ci_low'] = np.where(
            grouped['count'] >= 2,
            grouped['mean'] - 1.96 * grouped['sem'],
            grouped['mean']
        )
        grouped['ci_high'] = np.where(
            grouped['count'] >= 2,
            grouped['mean'] + 1.96 * grouped['sem'],
            grouped['mean']
        )
        
        # Get unique bins for coloring
        bins = sorted(grouped[bin_col].dropna().unique(), key=lambda x: str(x))
        
        # Plot each bin separately
        for bin_idx, bin_val in enumerate(bins):
            bin_data = grouped[grouped[bin_col] == bin_val].sort_values(event_time_col)
            if len(bin_data) == 0:
                continue
            
            tau = bin_data[event_time_col].astype(float).values
            mean_dose = bin_data["mean"].astype(float).values
            ci_low = bin_data["ci_low"].astype(float).values
            ci_high = bin_data["ci_high"].astype(float).values
            
            # Use palette index starting from 1 (0 is for pooled)
            color_idx = (bin_idx + 1) % len(palette)
            
            # Plot with error bars (like ATT^o by dose bin)
            for i in range(len(tau)):
                ax.errorbar(
                    [tau[i]], [mean_dose[i]],
                    yerr=[[mean_dose[i] - ci_low[i]], [ci_high[i] - mean_dose[i]]],
                    fmt="o", color=palette[color_idx], capsize=4, linewidth=2,
                    markersize=5
                )
            
            # Connect points with a line (with label for legend)
            ax.plot(tau, mean_dose, linewidth=1.5, color=palette[color_idx], 
                   alpha=0.5, zorder=0, label=str(bin_val))
        
        n_series = len(bins) + (1 if show_pooled else 0)
    else:
        n_series = 1 if show_pooled else 0
    
    # Show legend if we have multiple series (pooled + bins)
    if n_series > 1:
        ax.legend(loc='best', fontsize=10, frameon=False)
    
    # Reference τ = -1
    if ref_tau is not None:
        ax.axvline(float(ref_tau), color="0.2", linestyle="-", linewidth=1.2, alpha=0.7)
    
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
        ylabel="Effect (Δlog units)",
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
        es_df, ax=axes[2], title="Event study (pooled)", xlabel="Event time τ", ylabel=None
    )
    FIG.finalize(fig, axes, suptitle=suptitle, save=save, show=show)
    return fig