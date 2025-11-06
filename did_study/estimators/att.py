"""Pooled Average Treatment on the Treated (ATT^o) estimator.

Implements the collapsed‑regression estimator for ATT^o under staggered
adoption with continuous dose collapsed to a contemporaneous treated
indicator (absorbing). Returns coefficient, CRSE, analytic p, WCB p,
95% CI and MDE. Designed to match the design used elsewhere in the
package and to be reproducible via seed plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..helpers.config import StudyConfig
from ..helpers.utils import choose_wcb_weights_and_B
from ..robustness.stats.mde import analytic_mde_from_se
from ..robustness.wcb import wcb_att_pvalue_r


@dataclass
class AttResult:
    """Container for an estimated pooled treatment effect."""

    coef: float
    se: float
    p: float
    n_clusters: int
    n_obs: int
    p_wcb: float | np.nan
    lo: float
    hi: float
    mde: float


def estimate_att_o(panel: PanelData, config: StudyConfig) -> AttResult:
    """Estimate pooled ATT^o via collapsed regression with cluster‑robust SEs.

    Parameters
    ----------
    panel : PanelData
        Prepared panel with outcome, `treated_now`, covariates and FE identifiers.
    config : StudyConfig
        Includes FE flags, cluster id, bootstrap controls and seed.
    """
    df = panel.panel.copy()
    if df.empty:
        raise ValueError("Panel is empty; cannot estimate ATT.")

    cfg = config
    y = panel.outcome_name
    cluster_col = "unit_id"
    year_col = cfg.year_col
    covs: List[str] = panel.info.get("covariates_used", []) or []  # type: ignore

    rhs_terms = ["treated_now"] + covs + [f"C({year_col})"]
    include_unit_fe = not y.lower().startswith("d_")
    if include_unit_fe:
        rhs_terms.append(f"C({cluster_col})")

    formula = f"{y} ~ " + " + ".join(rhs_terms)
    required = [y, "treated_now", year_col, cluster_col] + covs
    used = df.dropna(subset=[c for c in required if c in df.columns]).copy()

    if used.empty or used[cluster_col].nunique() < 2:
        raise ValueError("Not enough usable data or clusters to estimate ATT.")

    groups = pd.factorize(used[cluster_col])[0]
    mod = smf.ols(formula, data=used).fit(cov_type="cluster", cov_kwds={"groups": groups})

    coef = float(mod.params.get("treated_now", np.nan))
    se = float(mod.bse.get("treated_now", np.nan))
    pval = float(mod.pvalues.get("treated_now", np.nan))
    n_clusters = int(used[cluster_col].nunique())
    n_obs = int(len(used))

    # t‑based 95% CI using G−1 dof
    df_eff = max(n_clusters - 1, 1)
    try:
        from scipy.stats import t as _t
        tcrit = _t.ppf(0.975, df_eff)
    except Exception:
        from statistics import NormalDist
        tcrit = NormalDist().inv_cdf(0.975)
    lo = coef - tcrit * se
    hi = coef + tcrit * se

    mde = analytic_mde_from_se(se, n_clusters)

    # Wild cluster bootstrap p‑value (R bridge)
    p_wcb: float | np.nan = np.nan
    if bool(getattr(cfg, "use_wcb", True)):
        wt, BB = choose_wcb_weights_and_B(n_clusters, getattr(cfg, "wcb_weights", None), getattr(cfg, "wcb_B", None))
        try:
            p_wcb = wcb_att_pvalue_r(
                used,
                outcome=y,
                regressors=["treated_now"] + covs,
                fe=[year_col] + ([cluster_col] if include_unit_fe else []),
                cluster=cluster_col,
                param="treated_now",
                B=int(BB),
                weights=wt,
                seed=int(getattr(cfg, "seed", 123) or 123),
                impose_null=True,
            )
        except Exception as e:  # pragma: no cover — robust to missing R
            print(f"[WCB] failed: {e}")
            p_wcb = np.nan

    return AttResult(
        coef=coef,
        se=se,
        p=pval,
        n_clusters=n_clusters,
        n_obs=n_obs,
        p_wcb=p_wcb,
        lo=lo,
        hi=hi,
        mde=mde,
    )
