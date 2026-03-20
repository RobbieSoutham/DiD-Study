# did_study/estimators/att_continuous.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..helpers.config import StudyConfig
from ..robustness.wcb import TestSpec, make_wcb_runner
from ..robustness.stats.mde import analytic_mde_from_se


@dataclass
class ContinuousAttResult:
    coef: float | np.nan
    se: float | np.nan
    p: float | np.nan
    p_wcb: float | np.nan
    model: Any
    used: pd.DataFrame
    dose_col: str = "dose_level"


def estimate_att_o_continuous(
    panel,
    config: StudyConfig,
    fe_terms: Optional[Sequence[str]] = None,
    *,
    dose_col: str = "dose_level",
    wcb_args: Optional[Dict[str, Any]] = None,
) -> ContinuousAttResult:
    """
    Linear-in-dose TWFE estimator.

    Regresses the outcome on dose_level (zero for controls and pre-treatment
    periods) with unit and time fixed effects. The coefficient captures the
    ATT per unit of dose (continuous treatment analogue of ATT^o).
    """
    df = panel.panel.copy()
    outcome = panel.outcome_name
    cluster_col = "unit_id"
    year_col = config.year_col

    if dose_col not in df.columns:
        return ContinuousAttResult(np.nan, np.nan, np.nan, np.nan, None, df, dose_col)

    include_unit_fe_default = not outcome.lower().startswith("d_")
    default_fe: List[str] = [year_col]
    if include_unit_fe_default:
        default_fe.insert(0, cluster_col)
    fe_terms = list(fe_terms) if fe_terms else default_fe

    covs = panel.info.get("covariates_used", []) or []
    need = [outcome, dose_col, cluster_col, year_col] + covs
    used = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    if used.empty or used[cluster_col].nunique() < 2:
        return ContinuousAttResult(np.nan, np.nan, np.nan, np.nan, None, used, dose_col)

    fe_rhs = [f"C({c})" for c in fe_terms]
    rhs = [dose_col] + covs + fe_rhs
    formula = f"{outcome} ~ " + " + ".join(rhs)

    groups = pd.factorize(used[cluster_col])[0]
    m = smf.ols(formula, data=used).fit(cov_type="cluster", cov_kwds={"groups": groups})
    coef = float(m.params.get(dose_col, np.nan))
    se = float(m.bse.get(dose_col, np.nan))
    p = float(m.pvalues.get(dose_col, np.nan))

    n_clusters = int(used[cluster_col].nunique())
    n_obs = int(len(used))
    mde_val = float("nan")
    if np.isfinite(se) and n_clusters > 1:
        try:
            mde_val = analytic_mde_from_se(se, n_clusters)
        except Exception:
            mde_val = float("nan")

    p_wcb = np.nan
    if wcb_args:
        try:
            base_terms = [dose_col] + covs
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

            candidate_cols = (
                [outcome, cluster_col, year_col]
                + base_terms
                + fe_terms
                + cluster_list
            )
            wcb_cols: List[str] = []
            for col in candidate_cols:
                if col in used.columns and col not in wcb_cols:
                    wcb_cols.append(col)
            wcb_df = used[wcb_cols].copy()

            runner = make_wcb_runner(
                df=wcb_df,
                outcome=outcome,
                regressors=base_terms,
                fe=list(fe_terms or []),
                cluster=cluster_list,
                B=B,
                weights=weights,
                impose_null=impose_null,
                seed=seed,
            )
            p_wcb = runner.pvalue(TestSpec(param=dose_col))
        except Exception as e:
            print("[WCB] failed; returning NaN. Reason:", repr(e))

    result = ContinuousAttResult(coef, se, p, p_wcb, m, used, dose_col)
    result.n_clusters = n_clusters
    result.clusters = n_clusters
    result.n_obs = n_obs
    result.n = n_obs
    result.mde = mde_val
    return result
