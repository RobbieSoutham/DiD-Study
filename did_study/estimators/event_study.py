from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from did_study.helpers.preparation import PanelData
from ..helpers.config import StudyConfig
from ..helpers.utils import log_wcb_call
from ..robustness.wcb import TestSpec, make_wcb_runner


@dataclass
class EventStudyResult:
    coefs: pd.DataFrame          # cols: event_time, beta, se, p
    pta_p: float | np.nan        # standard F-test across pre-treatment periods (leads <= -2)
    wcb_p: float | np.nan        # WCB joint p-value across all ES coefficients
    data: pd.DataFrame
    vcov: Any                    # ES-only vcov (pre then post)
    names_pre: List[str]
    names_post: List[str]


def _tau_from_name(col: str) -> Optional[int]:
    """Parse ES_t.. / ES_tm.. column names into integer event times."""
    if col.startswith("ES_tm"):
        return -int(col.replace("ES_tm", ""))
    if col.startswith("ES_t"):
        return int(col.replace("ES_t", ""))
    return None


def event_study(
    panel: PanelData,
    config: StudyConfig,
    fe_terms: Optional[Sequence[str]] = None,
    *,
    wcb_args: Optional[Dict[str, Any]] = None,
) -> EventStudyResult:
    df = panel.panel.copy()
    if df.empty:
        return EventStudyResult(
            coefs=pd.DataFrame(),
            pta_p=np.nan,
            wcb_p=np.nan,
            data=df,
            vcov=np.empty((0, 0)),
            names_pre=[],
            names_post=[],
        )

    outcome_col = panel.outcome_name
    year_col = config.year_col
    cluster_col = "unit_id"

    # Default FE: unit+year, unless outcome is first-difference
    include_unit_fe_default = not outcome_col.lower().startswith("d_")
    default_fe: List[str] = [year_col]
    if include_unit_fe_default:
        default_fe.insert(0, cluster_col)
    fe_terms = list(fe_terms) if fe_terms else default_fe

    pre = int(config.pre)
    post = int(config.post)
    min_cluster_support = int(config.min_cluster_support or 1)

    # ------------------------------------------------------------------
    # Build event-time dummies
    # ------------------------------------------------------------------
    df = df.copy()
    # continuous evt time
    df["evt"] = np.where(
        df["g"].notna(), (df[year_col] - df["g"]).astype(float), np.nan
    )

    inwin_any = df["evt"].between(-pre, post, inclusive="both")
    years_keep = df.loc[df["g"].notna() & inwin_any, year_col].unique()
    keep = df[df[year_col].isin(years_keep)].copy()
    keep["evt_int"] = keep["evt"].round().astype("Int64")

    sup = (
        keep[(keep["g"].notna()) & (keep["evt_int"].notna())]
        .groupby("evt_int")[cluster_col]
        .nunique()
        .rename("treated_clusters")
    )

    valid_times: List[int] = [
        int(t) for t, c in sup.items()
        if (-pre <= t <= post) and (c >= min_cluster_support)
    ]
    # drop t = -1 (reference)
    times_for_cols = sorted([t for t in valid_times if t != -1])

    if not times_for_cols:
        return EventStudyResult(
            coefs=pd.DataFrame(),
            pta_p=np.nan,
            wcb_p=np.nan,
            data=keep,
            vcov=np.empty((0, 0)),
            names_pre=[],
            names_post=[],
        )

    inter_cols: List[str] = []
    for t in times_for_cols:
        col = f"ES_tm{abs(t)}" if t < 0 else f"ES_t{t}"
        keep[col] = ((keep["evt_int"] == int(t)) & (keep["g"].notna())).astype(int)
        inter_cols.append(col)

    covs = panel.info.get("covariates_used", []) or []

    # ------------------------------------------------------------------
    # OLS with clustered SE
    # ------------------------------------------------------------------
    fe_rhs = [f"C({col})" for col in fe_terms]
    rhs_terms = inter_cols + covs + fe_rhs
    formula = f"{outcome_col} ~ " + " + ".join(rhs_terms)

    need_vars = [outcome_col, year_col, cluster_col] + inter_cols + covs
    used = keep.dropna(subset=[c for c in need_vars if c in keep.columns]).copy()
    if used.empty or used[cluster_col].nunique() < 2:
        return EventStudyResult(
            coefs=pd.DataFrame(),
            pta_p=np.nan,
            wcb_p=np.nan,
            data=used,
            vcov=np.empty((0, 0)),
            names_pre=[],
            names_post=[],
        )

    groups = pd.factorize(used[cluster_col])[0]
    m = smf.ols(formula, data=used).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )

    # ------------------------------------------------------------------
    # Coefficient table
    # ------------------------------------------------------------------
    rows: List[Dict[str, float]] = []
    for col in inter_cols:
        if col in m.params.index:
            tau = _tau_from_name(col)
            if tau is None:
                continue
            rows.append(
                {
                    "event_time": tau,
                    "beta": float(m.params[col]),
                    "se": float(m.bse[col]),
                    "p": float(m.pvalues[col]),
                }
            )
    coef_tab = pd.DataFrame(rows).sort_values("event_time")

    # ------------------------------------------------------------------
    # Tests:
    #   - pta_p: standard F-ztest of leads ≤ -2 (pre-trends)
    #   - wcb_p: WCB joint test of *all* ES coefficients (pooled experiment)
    # ------------------------------------------------------------------
    es_all_names: List[str] = [c for c in inter_cols if c in m.params.index]
    lead_names: List[str] = [
        c for c in es_all_names
        if (_tau_from_name(c) is not None and _tau_from_name(c) <= -2)
    ]

    # 1) PTA pretrend test: analytic F-test only
    def _f_test_or_nan(names: List[str]) -> float | np.nan:
        if not names:
            return np.nan
        H = " = 0, ".join(names) + " = 0"
        try:
            return float(m.f_test(H).pvalue)
        except Exception:
            return np.nan

    pta_p: float | np.nan = _f_test_or_nan(lead_names)

    # 2) Joint WCB test across all ES coefficients
    wcb_p: float | np.nan = np.nan
    if wcb_args and es_all_names:
        cluster_spec = (
            wcb_args.get("cluster_terms")
            or wcb_args.get("cluster_spec")
            or cluster_col
        )
        if isinstance(cluster_spec, str):
            cluster_list = [cluster_spec]
        else:
            cluster_list = list(cluster_spec or [])
        cluster_list = [c for c in cluster_list if c in used.columns]
        if not cluster_list:
            cluster_list = [cluster_col]

        impose_null = bool(wcb_args.get("impose_null", True))
        B = int(wcb_args.get("B", 9999))
        weights = wcb_args.get("weights", "rademacher")
        seed = wcb_args.get("seed")
        regressors = inter_cols + covs

        try:
            candidate_cols = (
                [outcome_col, cluster_col, year_col]
                + regressors
                + list(fe_terms or [])
                + cluster_list
            )
            wcb_cols: List[str] = []
            for col in candidate_cols:
                if col in used.columns and col not in wcb_cols:
                    wcb_cols.append(col)
            wcb_df = used[wcb_cols].copy()

            runner = make_wcb_runner(
                df=wcb_df,
                outcome=outcome_col,
                regressors=regressors,
                fe=list(fe_terms or []),
                cluster=cluster_list,
                B=B,
                weights=weights,
                impose_null=impose_null,
                seed=seed,
            )

            wcb_val = runner.pvalue(TestSpec(joint_zero=es_all_names))
            if isinstance(wcb_val, float) and not np.isnan(wcb_val):
                wcb_p = float(wcb_val)
            else:
                wcb_p = np.nan
        except Exception as e:
            # YOUR REQUEST: if WCB fails, print the exception
            print(f"[Event Study][joint WCB] failed: {e}")
            wcb_p = np.nan
            
    # ------------------------------------------------------------------
    # ES-only vcov aligned with betahat order: pre (<0 except -1) then post (>=0)
    # ------------------------------------------------------------------
    es_cov_full = m.cov_params()
    es_present = [c for c in inter_cols if c in es_cov_full.index]

    pre_names = sorted(
        [
            c
            for c in es_present
            if (_tau_from_name(c) is not None and _tau_from_name(c) < 0 and _tau_from_name(c) != -1)
        ],
        key=lambda x: _tau_from_name(x),
    )
    post_names = sorted(
        [
            c
            for c in es_present
            if (_tau_from_name(c) is not None and _tau_from_name(c) >= 0)
        ],
        key=lambda x: _tau_from_name(x),
    )
    ordered = pre_names + post_names
    es_vcov = (
        es_cov_full.loc[ordered, ordered].to_numpy(dtype=float)
        if ordered
        else np.empty((0, 0))
    )

    return EventStudyResult(
        coefs=coef_tab,
        pta_p=pta_p,
        wcb_p=wcb_p,
        data=used,
        vcov=es_vcov,
        names_pre=pre_names,
        names_post=post_names,
    )
