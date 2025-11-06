"""
Event study estimator for staggered adoption difference‑in‑differences.

This module implements an event study regression for panels prepared
by :mod:`did_study.helpers.preparation`.  It constructs interaction
terms between treatment status and event time and estimates the
dynamic treatment effects relative to a baseline period.  A joint
pre‑trend assessment tests whether all lead coefficients are zero.
Wild cluster bootstrap inference is available when requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..helpers.config import StudyConfig
from ..helpers.utils import choose_wcb_weights_and_B
from ..robustness.wcb import wcb_joint_pvalue_r


@dataclass
class EventStudyResult:
    """Container for event study estimates.

    Attributes
    ----------
    coefs : pandas.DataFrame
        Table of estimated event study coefficients.  Each row
        contains ``event_time``, ``beta``, ``se`` and ``p``.
    pta_p : float or numpy.nan
        P‑value for the joint test that all lead coefficients (t ≤ −2)
        equal zero.  ``np.nan`` if not computable.
    data : pandas.DataFrame
        The regression sample used for the estimation including the
        constructed indicator variables.  This can be used for
        diagnostic plots or further analysis.
    """

    coefs: pd.DataFrame
    pta_p: float | np.nan
    data: pd.DataFrame
    vcov: any


def event_study(panel: PanelData, config: StudyConfig) -> EventStudyResult:
    """Estimate an event study regression.

    Constructs event time indicators for each valid event time within
    ``[-pre, post]`` as specified in the configuration and regresses
    the outcome on these indicators, covariates and fixed effects.  A
    joint test is performed on the lead coefficients to assess
    parallel trends.  Wild cluster bootstrap inference is used when
    requested via ``config.use_wcb``.

    Parameters
    ----------
    panel : :class:`did_study.helpers.preparation.PanelData`
        Prepared panel with outcome, treatment timing and covariates.
    config : :class:`did_study.helpers.config.StudyConfig`
        Configuration specifying pre and post windows, covariates,
        bootstrap settings and minimum support.

    Returns
    -------
    :class:`EventStudyResult`
        Structured result containing the event study coefficients,
        pre‑trend assessment p‑value and regression sample.
    """
    df = panel.panel.copy()
    if df.empty:
        return EventStudyResult(coefs=pd.DataFrame(), pta_p=np.nan, data=df)
    cfg = config
    outcome_col = panel.outcome_name
    year_col = cfg.year_col
    cluster_col = "unit_id"
    pre = int(cfg.pre)
    post = int(cfg.post)
    min_cluster_support = int(cfg.min_cluster_support or 1)
    # compute event time: difference between year and cohort year g
    df = df.copy()
    df["evt"] = np.where(df["g"].notna(), (df[year_col] - df["g"]).astype(float), np.nan)
    # restrict to calendar years with at least one treated observation in window
    inwin_any = df["evt"].between(-pre, post, inclusive="both")
    years_keep = df.loc[df["g"].notna() & inwin_any, year_col].unique()
    keep = df[df[year_col].isin(years_keep)].copy()
    keep["evt_int"] = keep["evt"].round().astype("Int64")
    # compute support for each event time among ever treated units
    sup = (
        keep[(keep["g"].notna()) & (keep["evt_int"].notna())]
        .groupby("evt_int")[cluster_col]
        .nunique()
        .rename("treated_clusters")
    )
    valid_times: List[int] = []
    for t, c in sup.items():
        if (-pre <= t <= post) and (c >= min_cluster_support):
            valid_times.append(int(t))
    
    # omit baseline event time t = -1
    times_for_cols = sorted([t for t in valid_times if t != -1])
    if not times_for_cols:
        return EventStudyResult(coefs=pd.DataFrame(), pta_p=np.nan, data=keep)
    
    # create interaction columns
    inter_cols: List[str] = []
    for t in times_for_cols:
        if t < 0:
            col = f"ES_tm{abs(t)}"
        else:
            col = f"ES_t{t}"
        keep[col] = ((keep["evt_int"] == int(t)) & (keep["g"].notna())).astype(int)
        inter_cols.append(col)
    
    # collect covariates used
    covs = panel.info.get("covariates_used", []) or []  # type: ignore
    # build formula
    rhs_terms = inter_cols + covs + [f"C({year_col})"]
    if not outcome_col.lower().startswith("d_"):
        rhs_terms.append(f"C({cluster_col})")
    formula = f"{outcome_col} ~ " + " + ".join(rhs_terms)
    # drop missing rows
    need_vars = [outcome_col, year_col, cluster_col] + inter_cols + covs
    used = keep.dropna(subset=[c for c in need_vars if c in keep.columns]).copy()
    if used.empty or used[cluster_col].nunique() < 2:
        return EventStudyResult(coefs=pd.DataFrame(), pta_p=np.nan, data=used)
    
    # fit model
    groups = pd.factorize(used[cluster_col])[0]
    m = smf.ols(formula, data=used).fit(cov_type="cluster", cov_kwds={"groups": groups})
    
    # extract coefficients
    rows = []
    for col in inter_cols:
        if col in m.params.index:
            if col.startswith("ES_tm"):
                tau = -int(col.replace("ES_tm", ""))
            elif col.startswith("ES_t"):
                tau = int(col.replace("ES_t", ""))
            else:
                try:
                    tau = int(col.replace("ES_", ""))
                except Exception:
                    tau = 0
            rows.append({
                "event_time": tau,
                "beta": float(m.params[col]),
                "se": float(m.bse[col]),
                "p": float(m.pvalues[col]),
            })
    
    coef_tab = pd.DataFrame(rows).sort_values("event_time")
    
    # pre‑trend assessment: joint test of leads (t ≤ -2)
    lead_names: List[str] = []
    for col in inter_cols:
        # parse tau from column name
        if col.startswith("ES_tm"):
            tau = -int(col.replace("ES_tm", ""))
        elif col.startswith("ES_t"):
            tau = int(col.replace("ES_t", ""))
        else:
            try:
                tau = int(col.replace("ES_", ""))
            except Exception:
                tau = 0
        if tau <= -2:
            lead_names.append(col)
    
    pta_p = np.nan
    if lead_names:
        # We have lead coefficients - compute PTA
        if cfg.use_wcb:
            # bootstrap joint test
            try:
                n_clusters = int(used[cluster_col].nunique())
                wt, BB = choose_wcb_weights_and_B(n_clusters, None, cfg.wcb_B)
                fe = [year_col]
                if not outcome_col.lower().startswith("d_"):
                    fe.append(cluster_col)

                pta_p = wcb_joint_pvalue_r(
                    used,
                    outcome=outcome_col,
                    regressors=inter_cols + covs,
                    fe=fe,
                    cluster=cluster_col,
                    joint_zero=lead_names,
                    B=BB,
                    weights=wt,
                    impose_null=True,
                    seed=None,
                    clustid=cluster_col,
                )
                # If WCB returns NaN, fall back to analytic
                if isinstance(pta_p, float) and np.isnan(pta_p):
                    H = " = 0, ".join(lead_names) + " = 0"
                    pta_p = float(m.f_test(H).pvalue)
            except Exception as e:
                # If WCB fails, fall back to analytic Wald test
                H = " = 0, ".join(lead_names) + " = 0"
                pta_p = float(m.f_test(H).pvalue)
        else:
            # analytic Wald test using statsmodels f_test
            H = " = 0, ".join(lead_names) + " = 0"
            pta_p = float(m.f_test(H).pvalue)

        
    return EventStudyResult(
        coefs=coef_tab,
        pta_p=pta_p,
        data=used,
        vcov=m.cov_params()
        )
