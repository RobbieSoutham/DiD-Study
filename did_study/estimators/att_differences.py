# did_study/estimators/att_differences.py
# Callaway–Sant'Anna style ATT(g,t) using the Python `differences` package


from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from differences.attgt.attgt import ATTgt

import math
import os
import numpy as np
import pandas as pd


from ..helpers.config import StudyConfig
from ..helpers.preparation import PanelData

_WCB_VERBOSE = str(os.environ.get("DID_STUDY_WCB_VERBOSE", "0")).lower() in {"1", "true", "yes", "on"}
_DIFF_VERBOSE = str(os.environ.get("DID_STUDY_DIFF_VERBOSE", "0")).lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Optional import of `differences`
# ---------------------------------------------------------------------------


@dataclass
class DifferencesAttResult:
    """
    Container for results from the Callaway–Sant'Anna style estimator
    implemented in the Python ``differences`` package.


    - ``att_overall`` is the ATT^o summary parameter (simple aggregation).
    - ``se_overall`` and ``p_overall`` are its (preferably bootstrap)
      standard error and p-value.
    - ``se_overall_analytic`` / ``p_overall_analytic`` store the analytic
      counterparts when available.
    - ``se_overall_boot`` / ``p_overall_boot`` store bootstrap counterparts
      when available (these are what ``se_overall`` / ``p_overall`` default to).
    - ``used`` is the panel actually passed to ``differences``.
    - ``attgt_obj`` is the underlying ``ATTgt`` object (if available).
    - ``att_gt_df`` holds the ATT(g,t) table (whatever format
      ``ATTgt.results`` returns).
    """


    att_overall: float | np.nan
    se_overall: float | np.nan
    p_overall: float | np.nan
    used: pd.DataFrame
    attgt_obj: Any
    att_gt_df: pd.DataFrame


    # Extra detail fields (all NaN when not available)
    se_boot: float | np.nan = math.nan
    p_boot: float | np.nan = math.nan
    p_wcb: float | np.nan = math.nan

    ci_lower: float | np.nan = math.nan
    ci_upper: float | np.nan = math.nan



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _build_differences_input(panel: PanelData, config: StudyConfig) -> pd.DataFrame:
    """
    Prepare the panel in the canonical format expected by ``differences.ATTgt``.


    We follow the Quick Start in the package docs, which assumes a DataFrame
    with a 2-level (unit, time) index and at least:


        - ``cohort``: first treatment period (g_i), NaN for never-treated
        - ``y``: outcome variable


    plus any additional covariates used in the outcome regression.


    We treat ``treated_ever`` as the indicator for ever-treated units and
    ``g`` (constructed in :class:`PanelData`) as the cohort / first-treatment
    year when available.
    """
    df = panel.panel.copy()
    year_col = getattr(config, "year_col", "Year")
    outcome = getattr(panel, "outcome_name", None)


    # Basic sanity checks
    needed = ["unit_id", year_col, "treated_ever"]
    if "g" in df.columns:
        needed.append("g")
    if outcome:
        needed.append(outcome)


    present = [c for c in needed if c in df.columns]
    used = df#.dropna(subset=present).copy()
    used['never_treated'] = (used['treated_ever'] == 0).astype(int)


    if used.empty or used["unit_id"].nunique() < 2:
        raise ValueError(
            "[differences] Panel too small or ill-formed for ATTgt "
            f"(rows={len(used)}, units={used['unit_id'].nunique()})."
        )


    # 1) Canonical unit/time columns
    used["unit"] = used["unit_id"].astype(str)
    used["time"] = used[year_col].astype(int)


    # 2) Cohort (first treatment time); NaN for never-treated.
    #    This matches the behaviour in the R `did` package and the
    #    `differences` documentation.
    treated_mask = used["treated_ever"].astype(int) == 1


    if "g" in used.columns and used["g"].notna().any():
        # If we've already constructed a cohort variable in preparation,
        # reuse it (it should be the first treatment year).
        used.loc[treated_mask & used["g"].notna(), "cohort"] = (
            used.loc[treated_mask & used["g"].notna(), "g"].astype(float)
        )
    else:
        # Fallback: first time the unit is observed while treated_ever == 1
        first_treat = (
            used.loc[treated_mask]
                .groupby("unit")["time"]
                .transform("min")
        )
        used.loc[treated_mask, "cohort"] = first_treat


    # Never-treated units have cohort = NaN
    used.loc[~treated_mask, "cohort"] = np.nan


    # 3) Outcome as ``y``
    if outcome and outcome in used.columns:
        used["y"] = used[outcome].astype(float)
    else:
        # This should not happen in normal use – better to fail loudly.
        raise ValueError(
            f"[differences] Outcome column '{outcome}' not found in panel."
        )


    # 4) Cluster variable for `differences` – we cluster at the unit level.
    #    Keep it as an explicit column because ATTgt expects column names.
    used["cluster"] = used["unit"].astype(str)


    # 5) Final index
    used = used.sort_values(["unit", "time"]).set_index(["unit", "time"])


    return used



def _flatten_columns(df: pd.DataFrame) -> List[Tuple[str, ...]]:
    """
    Represent columns as tuples of strings regardless of whether they are
    simple Index or MultiIndex.
    """
    cols: List[Tuple[str, ...]] = []
    for col in df.columns:
        if isinstance(col, tuple):
            cols.append(tuple(str(c) for c in col))
        else:
            cols.append((str(col),))
    return cols



def _norm_pvalue(coef: float, se: float) -> float:
    if not np.isfinite(coef) or not np.isfinite(se) or se <= 0:
        return math.nan
    z = abs(coef / se)
    # 2 * (1 - Phi(z)) = erfc(z / sqrt(2))
    return math.erfc(z / math.sqrt(2.0))




def _extract_summary_from_agg(
    agg_df: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """
    Parse the output of ``ATTgt.aggregate(type_of_aggregation='simple', overall=True)``
    to extract ATT, SE, and confidence intervals.


    Returns:
        (att, se, ci_lower, ci_upper)
    """
    if not isinstance(agg_df, pd.DataFrame) or agg_df.empty:
        return math.nan, math.nan, math.nan, math.nan


    row = agg_df.iloc[0]
    col_tuples = _flatten_columns(agg_df)


    att = math.nan
    se = math.nan
    ci_lower = math.nan
    ci_upper = math.nan

    for tup, val in zip(col_tuples, row):
        last = tup[-1].lower()


        # ATT coefficient
        if last in ("att", "estimate", "effect", "coef"):
            att = float(val)
        # Standard error (prefer bootstrap, fallback to analytic)
        elif last in ("std_error", "std.error", "se"):
            se = float(val)
        # Confidence intervals
        elif last in ("lower", "lb", "ci_lower"):
            ci_lower = float(val)
        elif last in ("upper", "ub", "ci_upper"):
            ci_upper = float(val)


    return att, se, ci_lower, ci_upper



def _extract_bootstrap_details_from_full_results(
    res_df: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """
    Extract bootstrap SE and p-values from the FULL results dataframe
    (type_of_aggregation=None), which contains individual ATT(g,t) estimates.


    Returns:
        (se_analytic, p_analytic, se_boot, p_boot)
    """
    if not isinstance(res_df, pd.DataFrame) or res_df.empty:
        return math.nan, math.nan, math.nan, math.nan


    se_analytic = math.nan
    p_analytic = math.nan
    se_boot = math.nan
    p_boot = math.nan


    # Handle MultiIndex columns
    if isinstance(res_df.columns, pd.MultiIndex):
        # Try to extract analytic SE
        try:
            se_analytic = res_df[('ATTgtResult', 'analytic', 'std_error')].mean()
        except KeyError:
            pass


        # Try to extract bootstrap SE
        try:
            se_boot = res_df[('ATTgtResult', 'bootstrap', 'std_error')].mean()
        except KeyError:
            pass


        # Try to extract analytic p-value
        try:
            p_analytic = res_df[('ATTgtResult', 'analytic', 'pvalue')].mean()
        except KeyError:
            try:
                p_analytic = res_df[('ATTgtResult', 'analytic', 'p_value')].mean()
            except KeyError:
                pass


        # Try to extract bootstrap p-value
        try:
            p_boot = res_df[('ATTgtResult', 'bootstrap', 'pvalue')].mean()
        except KeyError:
            try:
                p_boot = res_df[('ATTgtResult', 'bootstrap', 'p_value')].mean()
            except KeyError:
                pass


    return se_analytic, p_analytic, se_boot, p_boot



# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------



def estimate_att_o_differences(
    panel: PanelData,
    config: StudyConfig,
    wcb_args: Optional[Dict[str, Any]] = None,  # kept for API symmetry; only B is used
) -> DifferencesAttResult:
    """
    Estimate ATT^o using the Callaway–Sant'Anna style ATT(g,t) implemented in
    the Python ``differences`` package.


    This is *purely optional* – if the package is not installed or fails for
    any reason, we return NaNs and allow the rest of the ``did_study`` pipeline
    to continue unaffected.


    Note on bootstrap / WCB
    ------------------------
    The current Python ``differences`` implementation exposes analytic standard
    errors and (ordinary) bootstrap options, but does *not* provide a wild
    cluster bootstrap for the aggregated ATT^o. When ``config.use_wcb`` or
    ``wcb_args`` are set, we pass the requested number of bootstrap iterations
    to ``ATTgt.fit`` (via its ``boot_iterations`` argument, which typically
    implements a cluster-robust bootstrap). This should not be interpreted as
    a full Rambachan–Roth style WCB – it is simply leveraging whatever
    bootstrap the package provides.
    """


    cfg = config
    used = _build_differences_input(panel, cfg)
    
    # 2) Construct ATTgt object
    ctor_kwargs: Dict[str, Any] = {
        "data": used,
        "cohort_name": "cohort",
    }
    if _DIFF_VERBOSE:
        print(f"[differences] ATTgt ctor kwargs: {ctor_kwargs}")
    attgt = ATTgt(**ctor_kwargs)  # type: ignore[arg-type]


    # 3) Fit ATT(g,t) with an outcome regression including covariates
    covariates = list(getattr(panel, "covariates", []) or [])
    covariates = [c for c in covariates if c in used.columns]


    formula_rhs = "1"
    if covariates:
        formula_rhs = "1 + " + " + ".join(covariates)
    fml = f"y ~ {formula_rhs}"
    if _DIFF_VERBOSE:
        print(f"[differences] Formula: {fml}")


    fit_kwargs: Dict[str, Any] = dict(
        random_state=getattr(config, "seed", None),
        progress_bar=_DIFF_VERBOSE,
        control_group="not_yet_treated",
        boot_iterations=(wcb_args or {}).get("B", 0),
        est_method='reg',
        #cluster_var=["cluster"],
        #est_method='dr'
        #cluster_var="unit_id",  # cluster at the unit level
    )


    attgt.fit(fml, **fit_kwargs)  # type: ignore[arg-type]


    # 4) Get FULL results first (contains bootstrap details)
    try:
        res_df = attgt.results(type_of_aggregation=None, overall=False)
        if isinstance(res_df, pd.DataFrame):
            att_gt_df = res_df.copy()
        else:
            att_gt_df = pd.DataFrame(res_df)
    except Exception as e:
        if _DIFF_VERBOSE:
            print(
                "[differences] ATTgt.results(type_of_aggregation=None, overall=False) "
                f"failed; Reason: {e!r}"
            )
        att_gt_df = pd.DataFrame()


    # 5) Extract bootstrap details from full results
    se_analytic, p_analytic, se_boot, p_boot = _extract_bootstrap_details_from_full_results(att_gt_df)


    # 6) Overall ATT^o via simple aggregation (for summary stats)
    agg_df = attgt.aggregate(
        type_of_aggregation="simple",
        overall=True,
        boot_iterations=(wcb_args).get("B", 0)
        )


    att_overall = se_overall = ci_lower = ci_upper = math.nan


    if isinstance(agg_df, pd.DataFrame) and not agg_df.empty:
        att_overall, se_overall, ci_lower, ci_upper = _extract_summary_from_agg(agg_df)
    
    
    # 7) Determine preferred SE and p-value (bootstrap > analytic)
    p_overall = _norm_pvalue(att_overall, se_overall)
    p_boot = _norm_pvalue(att_overall, se_boot)

    # Final container
    return DifferencesAttResult(
        att_overall=att_overall,
        se_overall=se_overall,
        p_overall=p_overall,
        
        used=used,
        attgt_obj=attgt,
        att_gt_df=att_gt_df,

        se_boot=se_boot,
        p_boot=p_boot,
        p_wcb=p_boot,

        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )
    
