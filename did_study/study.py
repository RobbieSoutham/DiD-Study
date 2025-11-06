# study.py
from __future__ import annotations
from typing import Any, Dict, Optional, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Package-local imports (match user's repo layout)
from .helpers.config import StudyConfig
from .helpers.preparation import PanelData
from .estimator import DidEstimator
from .robustness.honest_did import honest_did_bounds, make_M_grid
from .robustness.r_interface import wcb_joint_pvalue_r, wcb_equal_bins_pvalue_r
from .helpers.utils import choose_wcb_weights_and_B
from .estimators.bins import _bin_categorical, _build_bin_dummies


def _print_equation_py(
    title: str,
    outcome: str,
    regressors: Sequence[str],
    fe: Sequence[str],
    cluster: Optional[str],
) -> None:
    """
    Pretty-printer that mirrors fixest's y ~ x1 + x2 | fe1 + fe2 spec.
    NOTE: exactly ONE pipe, FE joined by '+'.
    """
    rhs = " + ".join(regressors) if regressors else "1"
    fe_txt = " + ".join([str(f).replace("|", "").strip() for f in fe if f]) if fe else "none"
    clu_txt = str(cluster) if cluster else "none"
    print(f"[PY/OLS] {title}\n  formula: {outcome} ~ {rhs} | {fe_txt}\n  cluster: {clu_txt}\n----------------------")


def _cat_with_order_from_edges(
    s_interval: pd.Series,
    *,
    untreated_label: str = "untreated",
    edges: Optional[Sequence[float]] = None,
    right_closed: bool = False,
) -> pd.Categorical:
    """
    Turn an Interval/float bin series into an ordered *string* categorical.

    If `edges` are provided, we build labels in ascending order based on those
    edges; otherwise fall back to the observed order with `untreated_label`
    first if present.
    """
    if edges is not None and len(edges) >= 2:
        # Recreate labels in order using pd.IntervalIndex to guarantee ordering
        bins = pd.IntervalIndex.from_breaks(edges, closed=("right" if right_closed else "left"))
        # Build a dummy Categorical with the intended order of string labels
        labels = [str(iv) for iv in bins]
        order = [untreated_label] + labels
        # Map original series to strings (NA -> untreated)
        as_str = s_interval.astype(object).where(s_interval.notna(), other=untreated_label).astype(str)
        return pd.Categorical(as_str, categories=order, ordered=True)
    # fallback: rely on helper (keeps untreated first if present)
    return _bin_categorical(s_interval, untreated_label=untreated_label, order=None)


class DidStudy:
    """
    Orchestrates the full study:
      0) Build panel
      1) Pooled ATT^O
      2) Binned ATT^O + joint tests across bins (WCB), analytic fallback
      3) Event study (pooled) + HonestDiD Δ^RM bounds

    Fixed effects policy (per best practice):
      • If `differenced = True` (first-differences outcome), include *time FE only*.
        Unit FE are absorbed by differencing and should not be added.
      • If `differenced = False`, include *unit FE + time FE*.

    Clustering: default to unit-level clusters (column 'unit_id'), unless the
    config exposes an explicit cluster column that exists in the working data.
    """

    def __init__(self, config: StudyConfig) -> None:
        self.config = config
        self.panel: Optional[PanelData] = None
        self.estimator = DidEstimator(self.config)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        # 0) Build panel
        self.panel = PanelData(self.config)
        panel = self.panel
        pdf = panel.panel.copy()

        y = panel.outcome_name
        year_col = self.config.year_col

        # FE policy: FD => time FE only; levels => unit + time FE
        differenced = bool(panel.info.get("differenced", getattr(self.config, "differenced", False)))
        include_unit_fe = not differenced

        # Controls (already engineered in PanelData)
        controls: List[str] = list(self.config.covariates or [])

        # Cluster column: prefer explicit config.cluster_col or fall back to unit_id
        cluster_col = getattr(self.config, "cluster_col", None) or "unit_id"
        if cluster_col not in pdf.columns:
            # fall back safely
            cluster_col = "unit_id"

        results: Dict[str, Any] = {
            "panel": pdf,
            "panel_info": panel.info,
            "config": self.config,
            "covariates_used": panel.info.get("covariates_used", []),
        }

        # ------------------------------------------------------------------
        # 1) Pooled ATT^O
        # ------------------------------------------------------------------
        att_pooled = self.estimator.estimate_att_o(panel)
        results["att_pooled"] = att_pooled

        # ------------------------------------------------------------------
        # 2) Binned ATT^O (and keep the sample used)
        # ------------------------------------------------------------------
        bins_res = self.estimator.estimate_binned_att_o(panel)  # BinAttResult
        # Expect: bins_res.table (summary), bins_res.used (df used for binned model)
        results["att_bins_table"] = getattr(bins_res, "table", None)
        g = getattr(bins_res, "used", None)
        if g is None or not isinstance(g, pd.DataFrame) or g.empty:
            results["att_bins_tests"] = {"allzero_p": float("nan"), "equal_p": float("nan"),
                                         "info": {"note": "no bins or empty sample"}}
            # Proceed to event study / HonestDiD anyway
        else:
            # Ensure the working frame has all necessary columns
            need_base = [y, year_col, "unit_id"] + controls
            g = g.dropna(subset=[c for c in need_base if c in g.columns]).copy()

            # Ensure a *string* categorical bin with 'untreated' first and positives ordered
            edges = (panel.info.get("dose_bin_edges") if isinstance(panel.info, dict) else None) or None
            right_closed = bool(getattr(self.config, "dose_bins_right_closed", False))
            cat_all = _cat_with_order_from_edges(
                g.get("dose_bin"),
                untreated_label="untreated",
                edges=edges,
                right_closed=right_closed,
            )
            g["dose_bin_cat"] = cat_all
            g["dose_bin_str"] = pd.Series(cat_all).astype(str)

            # Numeric bin dummies for WCB (baseline = 'untreated', dropped)
            Xb, bin_param_names, name_map = _build_bin_dummies(cat_all, drop_first=True, prefix="trbin")

            # Human-readable labels for positive (non-baseline) bins (same order as bin_param_names)
            inv_name_map = {v: k for k, v in name_map.items()}
            pos_labels_hr = [inv_name_map[c] for c in bin_param_names]

            # Assemble regressors matrix for WCB path
            regressors_eq: List[str] = []
            if Xb is not None and Xb.shape[1] > 0:
                for c in Xb.columns:
                    g[c] = Xb[c].astype(float)
                valid_bin_cols = [
                    c for c in Xb.columns
                    if c in g.columns
                    and pd.to_numeric(g[c], errors="coerce").var() > 0
                    and float(pd.to_numeric(g[c], errors="coerce").sum()) > 0
                ]
                regressors_eq.extend(valid_bin_cols)
                # Keep only valid bin parameter names/labels with variation
                bin_param_names = [c for c in bin_param_names if c in valid_bin_cols]
                pos_labels_hr = [inv_name_map[c] for c in bin_param_names]
            if controls:
                regressors_eq.extend(controls)

            # Deduplicate regressors while preserving order
            regressors_eq = list(dict.fromkeys(regressors_eq))
            bin_param_names = [c for c in bin_param_names if c in regressors_eq]
            pos_labels_hr = [inv_name_map[c] for c in bin_param_names]

            # FE list for R/feols
            fe_r: List[str] = [year_col] + (["unit_id"] if include_unit_fe else [])

            # Choose WCB settings
            G_total = int(pd.Series(g[cluster_col]).nunique()) if cluster_col in g.columns else 0
            wt, BB = choose_wcb_weights_and_B(G_total, self.config.wcb_B)

            # ---------------- Analytic omnibus OLS (all bins vs untreated) ----------------
            # Use Patsy with explicit FE as C(year) + C(unit) if needed
            present_bins = pd.unique(g["dose_bin_str"]) if "dose_bin_str" in g.columns else []
            baseline_ref = "untreated" if (isinstance(present_bins, np.ndarray) and ("untreated" in set(present_bins))) else (
                str(present_bins[0]) if len(present_bins) else "untreated"
            )
            rhs_terms = [f"C(dose_bin_str, Treatment(reference={baseline_ref!r}))"] + (controls or [])
            fe_terms = f" + C({year_col})" + (f" + C(unit_id)" if include_unit_fe else "")
            fml = f"{y} ~ {' + '.join(rhs_terms)}{fe_terms}"
            _print_equation_py("Omnibus/analytic", y, rhs_terms, [year_col] + (["unit_id"] if include_unit_fe else []), cluster_col)
            try:
                # Pre-drop rows with any NA in variables used by the model so groups match
                need_analytic = [y, "dose_bin_str"] + (controls or []) + [year_col] + (["unit_id"] if include_unit_fe else [])
                if cluster_col:
                    need_analytic += [cluster_col]
                g2 = g.dropna(subset=[c for c in need_analytic if c in g.columns]).copy()
                mod = smf.ols(fml, data=g2)
                if cluster_col in g2.columns:
                    _ = mod.fit(cov_type="cluster", cov_kwds={"groups": g2[cluster_col]})
                else:
                    _ = mod.fit()
            except Exception as e:
                print("[PY/OLS] analytic omnibus OLS fit failed:", repr(e))

            # ---------------- WCB joint tests ----------------
            p_wcb_allzero = float("nan")
            p_wcb_equal = float("nan")
            seed = getattr(self.config, "seed", None)
            try:
                if regressors_eq and bin_param_names:
                    # Prepare clean frame for R path: drop NAs and keep necessary columns only
                    need_r = [y] + list(regressors_eq) + list(fe_r) + ([cluster_col] if cluster_col else [])
                    g_r = g.dropna(subset=[c for c in need_r if c in g.columns]).copy()
                    if g_r.empty:
                        raise ValueError("No rows remain for WCB omnibus after dropna.")

                    # Remove duplicate columns that can confuse pandas dtype access
                    if g_r.columns.duplicated().any():
                        g_r = g_r.loc[:, ~g_r.columns.duplicated()].copy()

                    # Ensure numeric regressors are floats and retain only those with variation
                    for c in regressors_eq:
                        if c in g_r.columns:
                            g_r[c] = pd.to_numeric(g_r[c], errors="coerce")

                    active_bins: List[str] = []
                    for c in bin_param_names:
                        if c in g_r.columns:
                            g_r[c] = g_r[c].astype(float)
                            if g_r[c].var() > 0 and g_r[c].sum() > 0:
                                active_bins.append(c)

                    bin_param_names = [c for c in bin_param_names if c in active_bins]
                    pos_labels_hr = [inv_name_map[c] for c in bin_param_names]

                    if bin_param_names:
                        # H0: all positive-bin effects are zero (vs untreated baseline)
                        p_wcb_allzero = wcb_joint_pvalue_r(
                            g_r, outcome=y, regressors=regressors_eq, fe=fe_r, cluster=cluster_col,
                            joint_zero=bin_param_names, B=BB, weights=wt, impose_null=True, seed=seed,
                        )
                        # H0: all positive-bin effects are equal (pairwise diffs = 0)
                        p_wcb_equal = wcb_equal_bins_pvalue_r(
                            g_r, outcome=y, regressors=regressors_eq, fe=fe_r, cluster=cluster_col,
                            bin_params=bin_param_names, B=BB, weights=wt, impose_null=True, seed=seed,
                        )
            except Exception as e:
                print("[WCB] joint tests failed; returning NaN p-values. Reason:", repr(e))

            results["att_bins_used"] = g
            results["att_bins_labels"] = pos_labels_hr
            results["att_bins_param_names"] = bin_param_names
            results["att_bins_tests"] = {
                "allzero_p": float(p_wcb_allzero),
                "equal_p":   float(p_wcb_equal),
                "info": {"weights": wt, "B": BB, "cluster": cluster_col, "engine": "fwildclusterboot::boottest"},
            }

        # ------------------------------------------------------------------
        # 3) Event study (pooled) + HonestDiD (Δ^RM)
        # ------------------------------------------------------------------
        es = self.estimator.event_study(panel)  # EventStudyResult
        results["event_study"] = es

        # Split into pre/post by event_time (exclude τ = −1 baseline from pre)
        coef_tab = es.coefs.copy()
        if "event_time" not in coef_tab.columns:
            raise ValueError("event_study.coefs must include an 'event_time' column.")
        pre = coef_tab[(coef_tab["event_time"] < 0) & (coef_tab["event_time"] != -1)].sort_values("event_time")
        post = coef_tab[coef_tab["event_time"] >= 0].sort_values("event_time")

        betahat = np.concatenate([pre["beta"].to_numpy(), post["beta"].to_numpy()])
        sigma = getattr(es, "vcov", None)  # Optional vcov

        # l-weights: 'uniform' (default), 'last', 'post_ge_1'
        l_mode = getattr(self.config, "honestdid_l_weights", "uniform")
        l_vec = None
        if l_mode in ("last", "post_ge_1"):
            L = len(post)
            if l_mode == "last":
                lv = np.zeros(L, float); lv[-1] = 1.0
                l_vec = lv
            else:
                if L < 2:
                    l_vec = None  # will fall back to uniform inside bounds
                else:
                    lv = np.ones(L, float); lv[0] = 0.0
                    lv = lv / lv.sum()
                    l_vec = lv

        # Calibrate or use provided M grid
        Mbar = getattr(self.config, "honestdid_Mbar", None)
        grid = make_M_grid(Mbar)

        rows: List[Dict[str, float]] = []
        if len(post) >= 1 and len(pre) >= 1 and len(betahat) == (len(pre) + len(post)):
            for M in grid:
                b = honest_did_bounds(
                    betahat=betahat,
                    num_pre_periods=len(pre),
                    num_post_periods=len(post),
                    M=float(M),
                    bound_type="relative",
                    l_vec=l_vec,
                    sigma=sigma,
                    use_r=bool(getattr(self.config, "r_bridge", True) and sigma is not None),
                )
                rows.append({"M": float(M), "lower": b["lower"], "upper": b["upper"], "point": b["point"]})
        else:
            # If ES didn't produce usable pre/post, return empty but keep key
            rows = [{"M": float("nan"), "lower": float("nan"), "upper": float("nan"), "point": float("nan")}]

        results["honestdid"] = {"results_df": pd.DataFrame(rows), "theta_hat": float(rows[0]["point"])}

        return results
