# did_study/estimators/att.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Sequence
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from ..helpers.config import StudyConfig
from ..helpers.utils import log_wcb_call
from ..robustness.wcb import TestSpec, make_wcb_runner
from ..robustness.stats.mde import analytic_mde_from_se

@dataclass
class AttResult:
    coef: float | np.nan
    se: float | np.nan
    p: float | np.nan
    p_wcb: float | np.nan
    model: Any
    used: pd.DataFrame

def estimate_att_o(
    panel,
    config: StudyConfig,
    fe_terms: Optional[Sequence[str]] = None,
    *,
    wcb_args: Optional[Dict[str, Any]] = None,
) -> AttResult:
    # Implement two-period long-difference ATT^o (treated vs untreated)
    df = panel.panel.copy()
    outcome = panel.outcome_name
    cluster_col = "unit_id"
    year_col = config.year_col

    need_cols = [outcome, cluster_col, year_col, "treated_ever"]
    if any(c not in df.columns for c in need_cols):
        used_empty = df[[c for c in need_cols if c in df.columns]].copy()
        return AttResult(np.nan, np.nan, np.nan, np.nan, None, used_empty)

    pre_h = int(getattr(config, "pre", 3) or 3)
    post_h = int(getattr(config, "post", 5) or 5)

    treated_mask = (df.get("treated_ever", 0) == 1) & df.get("g").notna()
    treated_units = df.loc[treated_mask, [cluster_col, "g"]].drop_duplicates()

    pre_years: set[int] = set()
    post_years: set[int] = set()
    for _, row in treated_units.iterrows():
        g_i = int(row["g"])  # type: ignore[arg-type]
        pre_years.update(range(g_i - pre_h, g_i))
        post_years.update(range(g_i, g_i + post_h + 1))

    rows: List[Dict[str, Any]] = []
    for uid, d_u in df.groupby(cluster_col, sort=False):
        ever = int(d_u["treated_ever"].iloc[0]) if not d_u.empty else 0
        if ever == 1 and "g" in d_u.columns and d_u["g"].notna().any():
            g_i = int(d_u["g"].dropna().iloc[0])
            mask_pre = (d_u[year_col] >= (g_i - pre_h)) & (d_u[year_col] <= (g_i - 1))
            mask_post = (d_u[year_col] >= g_i) & (d_u[year_col] <= (g_i + post_h))
        else:
            if not pre_years or not post_years:
                continue
            mask_pre = d_u[year_col].isin(pre_years)
            mask_post = d_u[year_col].isin(post_years)

        y_pre = d_u.loc[mask_pre, outcome].dropna()
        y_post = d_u.loc[mask_post, outcome].dropna()
        if y_pre.empty or y_post.empty:
            continue
        delta = float(y_post.mean() - y_pre.mean())
        rows.append({cluster_col: uid, "treated_ever": ever, "delta_y": delta})

    collapsed = pd.DataFrame(rows)
    if collapsed.empty or collapsed[cluster_col].nunique() < 2:
        return AttResult(np.nan, np.nan, np.nan, np.nan, None, collapsed)

    try:
        groups = pd.factorize(collapsed[cluster_col])[0]
        m = smf.ols("delta_y ~ treated_ever", data=collapsed).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        coef = float(m.params.get("treated_ever", np.nan))
        se = float(m.bse.get("treated_ever", np.nan))
        p = float(m.pvalues.get("treated_ever", np.nan))
    except Exception:
        m = None
        coef = se = p = float("nan")

    n_clusters = int(collapsed[cluster_col].nunique())
    n_obs = int(len(collapsed))
    try:
        mde_val = analytic_mde_from_se(se, n_clusters) if np.isfinite(se) and n_clusters > 1 else float("nan")
    except Exception:
        mde_val = float("nan")

    result = AttResult(coef, se, p, np.nan, m, collapsed)
    result.n_clusters = n_clusters
    result.clusters = n_clusters
    result.n_obs = n_obs
    result.n = n_obs
    result.mde = mde_val
    return result
