# summary.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

# Local plotting helpers (centralized styling handled in plotting.FigFinalizer)
from .plotting import (
    FIG,
    plot_dose_distribution,
    plot_att_pooled_point,
    plot_att_combined,
    plot_event_study_line,
    plot_honest_did_M_curve,
    plot_support_by_tau,
    plot_mean_dose_by_tau,
    plot_att_combo,
)

# ================================
# Formatting helpers
# ================================

def _fmt_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return "NA"
    return f"{p:.3f}" if p >= 0.001 else "<0.001"

def _fmt_p_with_stars(p: Optional[float]) -> str:
    """Format p-value with asterisks for significance."""
    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return "NA"
    stars = ""
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
    return f"{p_str}{stars}"

def _ci_str(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"

def _percent(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}%"

def _actual_power_from_effect(effect: float,
                              se: float,
                              n_clusters: Optional[int],
                              *,
                              alpha: float = 0.05,
                              two_sided: bool = True) -> Optional[float]:
    """Approximate achieved power at the observed effect size.

    Mirrors the MDE formula used elsewhere: with df = n_clusters - 1 and
    a two-sided critical value, approximate power as CDF(delta - crit),
    where delta = |effect| / se. Falls back to a normal approximation if
    SciPy is not available.

    Returns None if inputs are not finite.
    """
    try:
        if not np.isfinite(effect) or not np.isfinite(se) or se == 0 or n_clusters is None:
            return None
        delta = abs(float(effect)) / abs(float(se))
        try:
            from scipy.stats import t  # type: ignore
            df = max(int(n_clusters) - 1, 1)
            crit = t.ppf(1 - alpha / 2, df) if two_sided else t.ppf(1 - alpha, df)
            power = float(t.cdf(delta - crit, df))
        except Exception:
            from statistics import NormalDist  # type: ignore
            nd = NormalDist()
            crit = nd.inv_cdf(1 - alpha / 2) if two_sided else nd.inv_cdf(1 - alpha)
            power = float(nd.cdf(delta - crit))
        return power
    except Exception:
        return None

def _rule(title: str | None = None) -> None:
    line = "=" * 78
    if title:
        print(f"\n{line}\n{title}\n{line}")
    else:
        print(f"\n{line}")


def _series_value_at(series: Any, idx: int) -> float:
    if series is None:
        return float("nan")
    try:
        if hasattr(series, "iloc"):
            return float(series.iloc[idx])
        return float(series[idx])
    except Exception:
        try:
            return float(list(series)[idx])
        except Exception:
            return float("nan")


def _bins_result_to_df(bin_result: Any) -> pd.DataFrame:
    if bin_result is None:
        return pd.DataFrame()
    names = list(getattr(bin_result, "bin_names", []) or [])
    if not names:
        return pd.DataFrame()

    coef_series = getattr(bin_result, "coef", None)
    se_series = getattr(bin_result, "se", None)
    p_series = getattr(bin_result, "p_value", None)
    p_wcb_series = getattr(bin_result, "p_value_wcb", None)
    n_treated = getattr(bin_result, "n_treated_bins", None)
    cluster_series = getattr(bin_result, "cluster_counts", None)
    mde_series = getattr(bin_result, "mde", None)

    rows = []
    for idx, name in enumerate(names):
        coef = _series_value_at(coef_series, idx)
        se = _series_value_at(se_series, idx)
        p = _series_value_at(p_series, idx)
        mde = _series_value_at(mde_series, idx)
        lo = coef - 1.96 * se if np.isfinite(se) else float("nan")
        hi = coef + 1.96 * se if np.isfinite(se) else float("nan")
        n_obs = None
        if isinstance(n_treated, pd.Series):
            n_obs = float(n_treated.get(name, np.nan))
        clusters = None
        if isinstance(cluster_series, pd.Series):
            clusters = float(cluster_series.get(name, np.nan))
        p_wcb = _series_value_at(p_wcb_series, idx) if p_wcb_series is not None else float("nan")

        rows.append(
            {
                "bin": name,
                "coef": coef,
                "se": se,
                "p": p,
                "p_wcb": p_wcb,
                "clusters": clusters,
                "n_obs": n_obs,
                "MDE_analytic": mde,
                "lo": lo,
                "hi": hi,
            }
        )

    return pd.DataFrame(rows)

# ================================
# Main printable carrier
# ================================

@dataclass
class StudyPrintable:
    """
    Minimal container the study runner can pass to summary() to print & plot.

    Expected keys (by convention of your pipeline):
      panel_info: dict with units, ever_treated, obs, post_rows, dose_bin_edges, dose_series
      att_pooled: dict with coef, se, p, lo, hi, clusters, n, p_wcb(optional), wcb_info(optional), mde(optional)
      att_bins: DataFrame with ['bin','coef','p','p_wcb','clusters','n_obs','MDE_analytic','lo','hi']
      es_pooled: DataFrame with ['event_time','beta','se'] or ['event_time','beta','lo','hi']
      pta_pooled_p: float
      es_wcb_p: Optional[float]  # joint WCB test across ES coefficients
      honestdid: dict with 'M','lo','hi','theta_hat' (optional)
      support_by_tau: DataFrame with ['event_time','units'] (optional)

      # Omnibus tests (WCB only):
      wcb_bins_allzero_p: Optional[float]  # H0: all positive-dose bin effects = 0
      wcb_bins_equal_p:  Optional[float]  # (optional) H0: all positive-dose bin effects are equal

      # Info for displaying WCB omnibus settings:
      wcb_bins_info: Optional[Dict[str, Any]]  # {'weights':..., 'B':..., 'cluster':...}

      panel_df: Optional[DataFrame] # panel data for diagnostic plots
    """
    panel_info: Dict[str, Any]
    att_pooled: Dict[str, Any]
    att_bins: pd.DataFrame
    es_pooled: pd.DataFrame
    pta_pooled_p: Optional[float]
    es_wcb_p: Optional[float] = None
    honest_did: Optional[Dict[str, Any]] = None
    support_by_tau: Optional[pd.DataFrame] = None

    # WCB omnibus tests only
    wcb_bins_allzero_p: Optional[float] = None
    wcb_bins_equal_p: Optional[float] = None
    wcb_bins_info: Optional[Dict[str, Any]] = None

    panel_df: Optional[pd.DataFrame] = None

# ================================
# Back-compat print blocks (still used internally)
# ================================

def print_panel_block(info: Dict[str, Any]) -> None:
    _rule("PANEL STATS")
    units = info.get("units")
    ever = info.get("ever_treated")
    obs = info.get("obs")
    post_rows = info.get("post_rows")
    post_share = (post_rows / max(obs, 1)) if obs else 0.0
    print(f"Units: {units} | Ever-treated: {ever} | Obs: {obs} | Post rows: {post_rows} ({_percent(post_share)})")

    edges = info.get("dose_bin_edges")
    support = info.get("dose_bin_counts")
    if edges is not None:
        print(f"Bin edges: {edges}")
    if support is not None:
        print("[Bin support] units/rows per bin:")
        for b, s in support.items():
            print(f"  - {b}: units={s.get('units', 'NA')} rows={s.get('rows', 'NA')}")

    # Quick histogram if dose_series present
    dose_series = info.get("dose_series")
    if dose_series is not None:
        plot_dose_distribution(
            dose_series,
            density=False
        )

def print_att_pooled_block(att: Dict[str, Any], att_bins: Optional[pd.DataFrame] = None) -> None:
    _rule("ATT^o (pooled)")
    coef = float(att["coef"])
    se = float(att.get("se", np.nan))
    lo = float(att.get("lo", coef - 1.96 * se))
    hi = float(att.get("hi", coef + 1.96 * se))
    p = att.get("p")
    p_wcb = att.get("p_wcb")
    clusters = att.get("clusters")
    n = att.get("n")
    mde = att.get("mde")
    se = att.get("se", np.nan)
    ncl = att.get("n_clusters", att.get("clusters", None))

    p_txt = _fmt_p(p)
    ci_txt = _ci_str(lo, hi)
    print(f"coef = {coef:.3f} (SE {se:.3f}, p {p_txt}) | 95% CI {ci_txt} | clusters={clusters}, n={n}")
    if p_wcb is not None:
        wcb_info = att.get("wcb_info", {})
        wtype = wcb_info.get("weights", "auto")
        B = wcb_info.get("B", "NA")
        print(f"WCB p = {_fmt_p(p_wcb)} - (fwildclusterboot::boottest, two-sided; weights={wtype}, B={B})")
    if mde is not None:
        print(f"MDE (alpha=0.05, 80% power): {mde:.4f} (Deltalog scale ~ percentage points; 0.01 ~ 1 p.p.)")

    # Plot combined: pooled + binned if available
    if att_bins is not None and len(att_bins) > 0:
        plot_att_combined(
            coef=coef, lo=lo, hi=hi,
            bins_df=att_bins,
            title="ATT$^{o}$ (pooled + by bin)",
            ylabel="Effect (Deltalog units)"
        )
    else:
        plot_att_pooled_point(coef=coef, lo=lo, hi=hi, title="ATT$^{o}$ (pooled)", ylabel="Effect (Deltalog units)")

def print_att_bins_block(
    tbl: pd.DataFrame,
    wcb_equal_p: Optional[float],
    wcb_allzero_p: Optional[float],
    *,
    show_equal: bool = False,
    wcb_info: Optional[Dict[str, Any]] = None
) -> None:
    _rule("ATT^o by dose bin")
    if tbl is None or len(tbl) == 0:
        print("(no bins)")
        return

    cols = ["bin", "coef", "p", "p_wcb", "clusters", "n_obs", "MDE_analytic", "lo", "hi"]
    cols = [c for c in cols if c in tbl.columns]
    disp = tbl[cols].copy()
    with pd.option_context("display.max_rows", 100, "display.width", 120):
        print(disp.to_string(index=False))

    # Omnibus tests: WCB-only
    if wcb_allzero_p is not None:
        w = (wcb_info or {}).get("weights", "auto")
        B = (wcb_info or {}).get("B", "NA")
        cl = (wcb_info or {}).get("cluster", "cluster")
        print(
            f"\n[WCB omnibus - H0: all bin effects = 0] "
            f"p = {_fmt_p(wcb_allzero_p)} "
            f"(fwildclusterboot::mboottest, two-sided; cluster={cl}; weights={w}, B={B})"
        )
    if show_equal and (wcb_equal_p is not None):
        w = (wcb_info or {}).get("weights", "auto")
        B = (wcb_info or {}).get("B", "NA")
        cl = (wcb_info or {}).get("cluster", "cluster")
        print(
            f"[WCB heterogeneity - H0: all bin effects equal] "
            f"p = {_fmt_p(wcb_equal_p)} "
            f"(fwildclusterboot::mboottest, two-sided; cluster={cl}; weights={w}, B={B})"
        )

def print_es_block(
    df_es: pd.DataFrame,
    pta_p: Optional[float],
    support_by_tau: Optional[pd.DataFrame] = None,
    panel_df: Optional[pd.DataFrame] = None,
    *,
    panel_info: Optional[Dict[str, Any]] = None,
) -> None:
    _rule("Event Study (pooled)")
    if df_es is None or len(df_es) == 0:
        print("(no event-study results)")
        return

    print("[first rows]")
    with pd.option_context("display.max_rows", 8, "display.width", 120):
        cols = [c for c in ["event_time", "beta", "se", "p", "lo", "hi"] if c in df_es.columns]
        print(df_es[cols].head(8).to_string(index=False))

    if pta_p is not None:
        verdict = "reject" if (pta_p is not None and pta_p < 0.05) else "fail to reject"
        print(f"PTA (leads joint = 0) p = {_fmt_p(pta_p)} - Pre-trend: {verdict} H0 at 5%.")

    # Restrict plotting to the configured pre/post window if provided
    df_plot = df_es.copy()
    if panel_info is not None and "pre" in panel_info and "post" in panel_info and "event_time" in df_plot.columns:
        try:
            pre = int(panel_info.get("pre", 0))
            post = int(panel_info.get("post", 0))
            df_plot = df_plot[(df_plot["event_time"] >= -pre) & (df_plot["event_time"] <= post)]
        except Exception:
            pass

    try:
        plot_event_study_line(df_plot, title="Event study (pooled)", xlabel="Event time tau", ylabel="Effect (Deltalog units)")
    except Exception as e:
        print(f"[plot warning] could not plot ES: {e}")

    if support_by_tau is not None and len(support_by_tau) > 0:
        try:
            plot_support_by_tau(support_by_tau, title="Pre-trend support (units by lead tau)", xlabel="Event time tau (leads)", ylabel="Units")
        except Exception as e:
            print(f"[plot warning] could not plot support-by-tau: {e}")

    # Plot mean dose by event time
    if panel_df is not None and "dose_level" in panel_df.columns and "event_time" in panel_df.columns:
        try:
            plot_mean_dose_by_tau(
                panel_df,
                title="Mean dose by event time",
                xlabel="Event time tau",
                ylabel="Mean dose",
            )
        except Exception as e:
            print(f"[plot warning] could not plot mean dose by tau: {e}")

def print_honestdid_block(hd: Optional[Dict[str, Any]]) -> None:
    _rule("HonestDiD")
    if not hd:
        print("(not computed)")
        return

    M = np.asarray(hd.get("M", []))
    lo = np.asarray(hd.get("lo", []))
    hi = np.asarray(hd.get("hi", []))
    theta_hat = hd.get("theta_hat", None)

    k = min(3, len(M))
    if k > 0:
        print("[first rows]")
        for i in range(k):
            th_txt = f"(naive theta_hat={theta_hat:+.3f})" if theta_hat is not None else ""
            print(f" M={M[i]:.2f}: [{lo[i]:+.3f}, {hi[i]:+.3f}] {th_txt}")

    include_zero = (lo <= 0) & (0 <= hi)
    if len(include_zero) > 0 and not include_zero.all():
        indices = np.where(~include_zero)[0]
        if len(indices) > 0:
            i0, i1 = indices.min(), indices.max()
            print(f" Summary: bounds exclude 0 for M in [{M[i0]:.2f}, {M[i1]:.2f}] (robust effect supported).")
    else:
        print(" Summary: bounds include 0 across M-grid (no robust sign).")

    try:
        plot_honest_did_M_curve(M, lo, hi, theta_hat=theta_hat, title="HonestDiD M-sensitivity", xlabel="M", ylabel="Bounded effect interval")
    except Exception as e:
        print(f"[plot warning] could not plot HonestDiD M-curve: {e}")

def print_overall_support(
    att: Dict[str, Any],
    pta_p: Optional[float],
    att_bins: Optional[pd.DataFrame] = None,
    wcb_allzero_p: Optional[float] = None,
    hd: Optional[Dict[str, Any]] = None,
    wcb_bins_info: Optional[Dict[str, Any]] = None,
    es_wcb_p: Optional[float] = None,
) -> None:
    _rule("Overall Support")

    # 1) PTA/Pre-trends
    if pta_p is not None:
        pta_pass = pta_p >= 0.05
        pta_stars = _fmt_p_with_stars(pta_p)
        print(f"PTA (pre-trends): p = {pta_stars} -> {'PASS' if pta_pass else 'FAIL'} (fail-to-reject is PASS).")
    else:
        print("PTA (pre-trends): NA")

    if es_wcb_p is not None:
        print(f"Event-study joint WCB test (all coefficients): p = {_fmt_p_with_stars(es_wcb_p)}")
    else:
        print("Event-study joint WCB test: NA")

    # 2) Main pooled estimate
    coef = att.get("coef", np.nan)
    p = att.get("p")
    pw = att.get("p_wcb")
    mde = att.get("mde")
    se = att.get("se", np.nan)
    ncl = att.get("n_clusters", att.get("clusters", None))

    p_stars = _fmt_p_with_stars(p)
    pw_stars = _fmt_p_with_stars(pw) if pw is not None else "NA"

    print(f"\nATT^o (pooled):")
    print(f"  Estimate: {coef:.4f} (p = {p_stars}, WCB p = {pw_stars})")

    # MDE comparison
    if mde is not None and not np.isnan(mde):
        larger_than_mde = abs(coef) > mde if not np.isnan(coef) else False
        robust_supported = larger_than_mde and (p is not None and p < 0.05)
        # MDE ratio (|estimate|/MDE)
        try:
            ratio = abs(float(coef)) / float(mde) if (np.isfinite(coef) and float(mde) > 0) else np.nan
        except Exception:
            ratio = np.nan
        if np.isfinite(ratio):
            print(f"  MDE ratio (|estimate|/MDE): {ratio:.2f}")

        # Actual power at observed |estimate|
        ap = _actual_power_from_effect(coef, se, ncl, alpha=0.05, two_sided=True)
        if ap is not None and np.isfinite(ap):
            print(f"  Actual power @ |estimate|: {_percent(ap, digits=1)}")
        print(f"  MDE (alpha=0.05, 80% power): {mde:.4f}")
        print(f"  |estimate| > MDE: {'Yes' if larger_than_mde else 'No'} -> {'Robust effect supported' if robust_supported else 'Effect may not be robust'}")
    elif mde is not None:
        print(f"  MDE: {mde:.4f}")

    # 3) Binned estimates (per-bin display)
    if att_bins is not None and len(att_bins) > 0:
        print(f"\nATT^o by bin:")
        for _, row in att_bins.iterrows():
            bin_name = str(row.get("bin", "?"))
            bin_coef = float(row.get("coef", np.nan))
            bin_p = row.get("p")
            bin_pw = row.get("p_wcb")
            bin_p_stars = _fmt_p_with_stars(bin_p)
            bin_pw_stars = _fmt_p_with_stars(bin_pw) if (bin_pw is not None and not (isinstance(bin_pw, float) and np.isnan(bin_pw))) else "NA"
            print(f"  {bin_name}: {bin_coef:.4f} (p = {bin_p_stars}, WCB p = {bin_pw_stars})")

    # 4) WCB omnibus test (bins)
    if wcb_allzero_p is not None:
        w = (wcb_bins_info or {}).get("weights", "auto")
        B = (wcb_bins_info or {}).get("B", "NA")
        cl = (wcb_bins_info or {}).get("cluster", "cluster")
        wcb_joint_stars = _fmt_p_with_stars(wcb_allzero_p)
        print(
            f"\nWCB omnibus over bins - H0: all positive-dose bin effects = 0 "
            f"(fwildclusterboot::mboottest, two-sided; cluster={cl}; weights={w}, B={B}): "
            f"p = {wcb_joint_stars}"
        )
    else:
        print("\nWCB omnibus over bins: NA (not computed)")

    # 5) HonestDiD robustness
    if hd and "M" in hd and "lo" in hd and "hi" in hd:
        lo = np.asarray(hd["lo"])
        hi = np.asarray(hd["hi"])
        M = np.asarray(hd["M"])
        include_zero = (lo <= 0) & (0 <= hi)
        if len(include_zero) > 0 and not include_zero.all():
            indices = np.where(~include_zero)[0]
            if len(indices) > 0:
                i0, i1 = indices.min(), indices.max()
                print(f"\nHonestDiD: bounds exclude 0 for M in [{M[i0]:.2f}, {M[i1]:.2f}] -> robust effect supported.")
            else:
                print(f"\nHonestDiD: bounds include 0 across M-grid -> no robust sign.")
        else:
            print(f"\nHonestDiD: bounds include 0 across M-grid -> no robust sign.")

# ================================
# New: unified ATT summary table (pooled + per-bin)
# ================================

def _build_att_summary_table(att_pooled: Dict[str, Any], att_bins: pd.DataFrame) -> pd.DataFrame:
    """
    Create a single table with the pooled ATT^o and each dose bin on separate rows.
    Columns: ['group','coef','se','p','95% CI','WCB p'] with stars on p and WCB p.
    """
    rows: list[dict[str, Any]] = []

    # Pooled row
    coef = float(att_pooled.get("coef", np.nan))
    se = float(att_pooled.get("se", np.nan)) if att_pooled.get("se", None) is not None else np.nan
    lo = float(att_pooled.get("lo", coef - (1.96 * se if np.isfinite(se) else 0.0)))
    hi = float(att_pooled.get("hi", coef + (1.96 * se if np.isfinite(se) else 0.0)))
    p = att_pooled.get("p")
    pw = att_pooled.get("p_wcb")

    rows.append({
        "group": "pooled",
        "coef": f"{coef:.3f}",
        "se": f"{se:.3f}" if np.isfinite(se) else "NA",
        "p": _fmt_p_with_stars(p),
        "95% CI": _ci_str(lo, hi),
        "WCB p": _fmt_p_with_stars(pw) if pw is not None else "NA",
    })

    # Per-bin rows
    if isinstance(att_bins, pd.DataFrame) and len(att_bins) > 0:
        bins_df = att_bins.copy()
        if "bin_label" in bins_df.columns and "bin" not in bins_df.columns:
            bins_df = bins_df.rename(columns={"bin_label": "bin"})
        for _, r in bins_df.iterrows():
            bname = str(r.get("bin", "?"))
            bcoef = float(r.get("coef", np.nan))
            bse = r.get("se", np.nan)
            # compute CI if needed
            blo = r.get("lo", None)
            bhi = r.get("hi", None)
            if (blo is None or bhi is None) and np.isfinite(bse):
                blo = bcoef - 1.96 * float(bse)
                bhi = bcoef + 1.96 * float(bse)
            # fetch p / wcb
            bp = r.get("p", None)
            bpw = r.get("p_wcb", None)

            rows.append({
                "group": bname,
                "coef": f"{bcoef:.3f}" if np.isfinite(bcoef) else "NA",
                "se": f"{float(bse):.3f}" if (bse is not None and np.isfinite(float(bse))) else "NA",
                "p": _fmt_p_with_stars(bp),
                "95% CI": _ci_str(float(blo), float(bhi)) if (blo is not None and bhi is not None) else "NA",
                "WCB p": _fmt_p_with_stars(bpw) if bpw is not None else "NA",
            })

    return pd.DataFrame(rows, columns=["group", "coef", "se", "p", "95% CI", "WCB p"])

def print_att_summary_table(att_pooled: Dict[str, Any], att_bins: pd.DataFrame) -> None:
    _rule("ATT^o summary table (pooled + bins)")
    tbl = _build_att_summary_table(att_pooled, att_bins)
    with pd.option_context("display.max_rows", 200, "display.max_colwidth", 120, "display.width", 120):
        print(tbl.to_string(index=False))

# ================================
# Adapter: DidStudy.run() -> StudyPrintable
# ================================

def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    out = {}
    for k in ["coef","se","p","p_wcb","lo","hi","mde","n_clusters","n_obs","clusters","n"]:
        if hasattr(obj, k):
            out[k] = getattr(obj, k)
    used = getattr(obj, "used", None)
    if isinstance(used, pd.DataFrame):
        out.setdefault("n_obs", len(used))
        if "unit_id" in used.columns:
            clusters = used["unit_id"].nunique()
            out.setdefault("n_clusters", clusters)
            out.setdefault("clusters", clusters)
    return out

def study_printable_from_didstudy(results: Dict[str, Any]) -> StudyPrintable:
    def _res_get(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _res_pick(obj: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            if not name:
                continue
            val = _res_get(obj, name, None)
            if val is not None:
                return val
        return default

    data_obj = _res_pick(results, "data")
    panel_info = _res_pick(results, "panel_info") or {}
    if not panel_info and data_obj is not None:
        panel_info = getattr(data_obj, "info", {}) or {}
    if not isinstance(panel_info, dict):
        panel_info = dict(panel_info)

    panel_df = _res_pick(results, "panel")
    if not isinstance(panel_df, pd.DataFrame) and data_obj is not None:
        panel_df = getattr(data_obj, "panel", None)
    if not isinstance(panel_df, pd.DataFrame):
        panel_df = None

    # --- Fallbacks so PANEL STATS always prints real numbers ---
    if panel_df is not None:
        panel_info.setdefault("units", int(panel_df["unit_id"].nunique()) if "unit_id" in panel_df.columns else None)
        if "treated_ever" in panel_df.columns:
            try:
                ever = int(panel_df.groupby("unit_id")["treated_ever"].max().sum())
            except Exception:
                ever = int(panel_df["treated_ever"].max()) if "treated_ever" in panel_df.columns else None
            panel_info.setdefault("ever_treated", ever)
        panel_info.setdefault("obs", int(len(panel_df)))
        if "post" in panel_df.columns:
            panel_info.setdefault("post_rows", int((panel_df["post"] == 1).sum()))

        if "dose_bin_counts" not in panel_info and {"dose_bin", "unit_id", "post"} <= set(panel_df.columns):
            post = panel_df.loc[panel_df["post"] == 1].copy()
            if not post.empty:
                counts = {}
                for b, g in post.groupby("dose_bin", dropna=True):
                    counts[str(b)] = {"units": int(g["unit_id"].nunique()), "rows": int(len(g))}
                if counts:
                    panel_info["dose_bin_counts"] = counts

        if "dose_series" not in panel_info and "dose_level" in panel_df.columns:
            panel_info["dose_series"] = panel_df["dose_level"].astype(float).dropna()

    att_obj = _res_pick(results, "att", "att_pooled")
    att_d = _as_dict(att_obj)

    wcb_info = _res_pick(results, "wcb_info")
    if wcb_info is None:
        wcb_meta = _res_pick(results, "wcb_meta")
        wcb_info = wcb_meta if isinstance(wcb_meta, dict) else {}
    config = _res_pick(results, "config")
    if hasattr(att_obj, "p_wcb") and getattr(att_obj, "p_wcb") is not None and config:
        wcb_info = {
            "weights": getattr(config, "wcb_weights", "auto"),
            "B": getattr(config, "wcb_B", 9999),
        }

    coef = float(att_d.get("coef", np.nan))
    se_raw = att_d.get("se", np.nan)
    se = float(se_raw) if se_raw is not None else np.nan

    def _filled_ci(side_val: Any, fallback: float) -> float:
        try:
            val = float(side_val)
        except (TypeError, ValueError):
            val = np.nan
        if np.isnan(val) and np.isfinite(fallback):
            return fallback
        return val

    lo = _filled_ci(att_d.get("lo", np.nan), coef - 1.96 * se if np.isfinite(se) else np.nan)
    hi = _filled_ci(att_d.get("hi", np.nan), coef + 1.96 * se if np.isfinite(se) else np.nan)

    att_pooled = {
        "coef": coef,
        "se": se,
        "p": float(att_d.get("p", np.nan)),
        "lo": lo,
        "hi": hi,
        "p_wcb": None if (att_d.get("p_wcb") is None or (isinstance(att_d.get("p_wcb"), float) and np.isnan(att_d.get("p_wcb")))) else float(att_d.get("p_wcb")),
        "clusters": int(att_d.get("n_clusters", att_d.get("clusters", np.nan))) if att_d.get("n_clusters", None) is not None or att_d.get("clusters", None) is not None else None,
        "n": int(att_d.get("n_obs", att_d.get("n", np.nan))) if att_d.get("n_obs", None) is not None or att_d.get("n", None) is not None else None,
        "mde": float(att_d.get("mde", np.nan)) if att_d.get("mde", None) is not None else None,
        "wcb_info": wcb_info,
    }

    att_bins = _res_pick(results, "att_bins", "bins")
    if isinstance(att_bins, pd.DataFrame):
        bins_df = att_bins.copy()
    else:
        bins_df = _bins_result_to_df(att_bins)
    ren = {}
    if "bin_label" in bins_df.columns and "bin" not in bins_df.columns:
        ren["bin_label"] = "bin"
    if "SE" in bins_df.columns and "se" not in bins_df.columns:
        ren["SE"] = "se"
    if ren:
        bins_df = bins_df.rename(columns=ren)

    es_wcb = _res_pick(results, "es_wcb_p")
    es_obj = _res_pick(results, "event_study", "es_pooled")
    if es_obj is None:
        es_df = pd.DataFrame()
        pta_p = None
    else:
        if hasattr(es_obj, "coefs"):
            es_df = getattr(es_obj, "coefs")
            pta_p_raw = getattr(es_obj, "pta_p", np.nan)
            pta_p = None if (isinstance(pta_p_raw, float) and np.isnan(pta_p_raw)) else pta_p_raw
            es_wcb = getattr(es_obj, "wcb_p", None)
        elif isinstance(es_obj, dict):
            es_df = es_obj.get("coefs", pd.DataFrame())
            pta_p_raw = es_obj.get("pta_p", np.nan)
            pta_p = None if (isinstance(pta_p_raw, float) and np.isnan(pta_p_raw)) else pta_p_raw
            es_wcb = es_obj.get("wcb_p")
        else:
            es_df = pd.DataFrame()
            pta_p = None
        if "beta" in es_df.columns and ("lo" not in es_df.columns or "hi" not in es_df.columns):
            if "se" in es_df.columns:
                se = es_df["se"].astype(float)
                es_df = es_df.assign(lo=es_df["beta"].astype(float) - 1.96 * se, hi=es_df["beta"].astype(float) + 1.96 * se)
        if isinstance(es_wcb, float) and np.isnan(es_wcb):
            es_wcb = None

    honest = getattr(results, "honest_did", None)

    support_df = _res_pick(results, "support_by_tau")
    if not isinstance(support_df, pd.DataFrame):
        support_df = None

    tests = _res_pick(results, "att_bins_tests") or {}
    wcb_allzero_p = tests.get("allzero_p")
    wcb_equal_p = tests.get("equal_p")
    wcb_bins_info = tests.get("info")

    if wcb_bins_info is None and config is not None:
        wcb_bins_info = {
            "weights": getattr(config, "wcb_weights", "auto"),
            "B": getattr(config, "wcb_B", 9999),
            "cluster": getattr(config, "cluster_col", "cluster"),
            "engine": "fwildclusterboot::mboottest" if getattr(config, "use_wcb", False) else "statsmodels.wald_test",
            "method": "wcb" if getattr(config, "use_wcb", False) else "analytic",
        }

    return StudyPrintable(
        panel_info=panel_info,
        att_pooled=att_pooled,
        att_bins=bins_df,
        es_pooled=es_df,
        pta_pooled_p=pta_p,
        es_wcb_p=es_wcb,
        honest_did=honest,
        support_by_tau=support_df,
        wcb_bins_allzero_p=wcb_allzero_p,
        wcb_bins_equal_p=wcb_equal_p,
        wcb_bins_info=wcb_bins_info,
        panel_df=panel_df,
    )

# ================================
# Back-compat entry points
# ================================

def print_and_plot_summary(
    sp: StudyPrintable,
    *,
    make_combo: bool = False,
    show_equal_bins_test: bool = False
) -> None:
    """One-call pretty summary with plots."""
    print_panel_block(sp.panel_info)
    print_att_pooled_block(sp.att_pooled, sp.att_bins)
    print_att_bins_block(
        sp.att_bins,
        sp.wcb_bins_equal_p,
        sp.wcb_bins_allzero_p,
        show_equal=show_equal_bins_test,
        wcb_info=sp.wcb_bins_info
    )
    # Unified table:
    print_att_summary_table(sp.att_pooled, sp.att_bins)
    print_es_block(sp.es_pooled, sp.pta_pooled_p, sp.support_by_tau, sp.panel_df, panel_info=sp.panel_info)
    print_honestdid_block(sp.honest_did)
    print_overall_support(
        sp.att_pooled,
        sp.pta_pooled_p,
        sp.att_bins,
        sp.wcb_bins_allzero_p,
        sp.honest_did,
        wcb_bins_info=sp.wcb_bins_info,
        es_wcb_p=sp.es_wcb_p,
    )

    if make_combo and sp.att_bins is not None and len(sp.att_bins) > 0:
        pooled = {
            "coef": sp.att_pooled["coef"],
            "lo": sp.att_pooled.get("lo", sp.att_pooled["coef"] - 1.96 * sp.att_pooled.get("se", 0.0)),
            "hi": sp.att_pooled.get("hi", sp.att_pooled["coef"] + 1.96 * sp.att_pooled.get("se", 0.0)),
        }
        binned_df = sp.att_bins[["bin", "coef", "lo", "hi"]].copy()
        es_df = sp.es_pooled.copy()
        plot_att_combo(pooled, binned_df, es_df, suptitle="ATT$^{o}$ summary")

# ================================
# Reporter layer (optional)
# ================================

@dataclass
class StudyReportOptions:
    """Display options for the reporter layer."""
    show_equal_bins_test: bool = False   # optional WCB equal-effects test
    show_combo_figure: bool = False      # optional composite figure
    section_titles: bool = True          # print section rules/titles

class _BaseSection:
    def __init__(self, opts: StudyReportOptions):
        self.opts = opts
    def _title(self, t: str) -> None:
        if self.opts.section_titles:
            _rule(t)

class PanelReporter(_BaseSection):
    def __init__(self, panel_info: Dict[str, Any], opts: StudyReportOptions):
        super().__init__(opts)
        self.info = panel_info
    def show(self) -> None:
        print_panel_block(self.info)

class ATTReporter(_BaseSection):
    def __init__(self, att_pooled: Dict[str, Any], att_bins: pd.DataFrame, opts: StudyReportOptions):
        super().__init__(opts)
        self.att = att_pooled
        self.att_bins = att_bins
    def show(self) -> None:
        print_att_pooled_block(self.att, self.att_bins)

class BinsReporter(_BaseSection):
    def __init__(self, att_bins: pd.DataFrame, allzero_p: Optional[float], equal_p: Optional[float], info: Optional[Dict[str, Any]], opts: StudyReportOptions):
        super().__init__(opts)
        self.tbl = att_bins
        self.allzero = allzero_p
        self.equal = equal_p
        self.info = info
    def show(self) -> None:
        print_att_bins_block(self.tbl, self.equal, self.allzero, show_equal=self.opts.show_equal_bins_test, wcb_info=self.info)

class ESReporter(_BaseSection):
    def __init__(self, es_df: pd.DataFrame, pta_p: Optional[float], support_by_tau: Optional[pd.DataFrame], panel_df: Optional[pd.DataFrame], opts: StudyReportOptions):
        super().__init__(opts)
        self.es_df = es_df
        self.pta_p = pta_p
        self.support = support_by_tau
        self.panel_df = panel_df
    def show(self) -> None:
        print_es_block(self.es_df, self.pta_p, self.support, self.panel_df, panel_info=self.info)

class HonestDidReporter(_BaseSection):
    def __init__(self, hd: Optional[Dict[str, Any]], opts: StudyReportOptions):
        super().__init__(opts)
        self.hd = hd
    def show(self) -> None:
        print_honestdid_block(self.hd)

class StudyReporter:
    """
    High-level reporter that renders a DidStudy run in clearly separated blocks,
    with consistent styling and optional plots.

    Usage:
        results = DidStudy(cfg).run()
        rep = StudyReporter.from_didstudy(results, opts=StudyReportOptions(show_combo_figure=True))
        rep.show_all()
    """
    def __init__(self, printable: StudyPrintable, opts: Optional[StudyReportOptions] = None) -> None:
        self.sp = printable
        self.opts = opts or StudyReportOptions()

        # compose sections
        self.panel = PanelReporter(self.sp.panel_info, self.opts)
        self.att = ATTReporter(self.sp.att_pooled, self.sp.att_bins, self.opts)
        self.bins = BinsReporter(self.sp.att_bins, self.sp.wcb_bins_allzero_p, self.sp.wcb_bins_equal_p, self.sp.wcb_bins_info, self.opts)
        self.es = ESReporter(self.sp.es_pooled, self.sp.pta_pooled_p, self.sp.support_by_tau, self.sp.panel_df, self.opts)
        self.honest = HonestDidReporter(self.sp.honest_did, self.opts)

    @classmethod
    def from_didstudy(cls, results: Dict[str, Any], opts: Optional[StudyReportOptions] = None) -> "StudyReporter":
        return cls(study_printable_from_didstudy(results), opts=opts)

    def show_overview(self) -> None:
        self.panel.show()

    def show_att_pooled(self) -> None:
        self.att.show()

    def show_att_bins(self) -> None:
        self.bins.show()

    def show_event_study(self) -> None:
        self.es.show()

    def show_honestdid(self) -> None:
        self.honest.show()

    def show_all(self) -> None:
        self.show_overview()
        self.show_att_pooled()
        self.show_att_bins()
        # Unified table:
        print_att_summary_table(self.sp.att_pooled, self.sp.att_bins)
        self.show_event_study()
        self.show_honestdid()
        print_overall_support(
            self.sp.att_pooled,
            self.sp.pta_pooled_p,
            self.sp.att_bins,
            self.sp.wcb_bins_allzero_p,
            self.sp.honest_did,
            wcb_bins_info=self.sp.wcb_bins_info,
            es_wcb_p=self.sp.es_wcb_p,
        )
        # Optional super-figure (pooled + by-bin + ES)
        if self.opts.show_combo_figure and self.sp.att_bins is not None and len(self.sp.att_bins) > 0:
            pooled = {
                "coef": self.sp.att_pooled["coef"],
                "lo": self.sp.att_pooled.get("lo", self.sp.att_pooled["coef"] - 1.96 * self.sp.att_pooled.get("se", 0.0)),
                "hi": self.sp.att_pooled.get("hi", self.sp.att_pooled["coef"] + 1.96 * self.sp.att_pooled.get("se", 0.0)),
            }
            binned_df = self.sp.att_bins[["bin", "coef", "lo", "hi"]].copy()
            es_df = self.sp.es_pooled.copy()
            plot_att_combo(pooled, binned_df, es_df, suptitle="ATT$^{o}$ summary")
