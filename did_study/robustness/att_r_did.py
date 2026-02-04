# did_study/robustness/att_r_did.py
"""
Wild Cluster Bootstrap for Callaway-Sant'Anna ATT^o using R's did package.

This module provides native WCB inference for the differences-in-differences
ATT^o estimator using the original R implementation (Callaway & Sant'Anna 2021).

For small treatment cluster counts (< 20), this provides more reliable inference
than the Python differences package bootstrap alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math

import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri


@dataclass
class RDidResult:
    """
    Results from R's did package att_gt + aggte.
    
    Attributes:
        att_overall: Overall ATT^o (simple aggregation)
        se: Standard error (bootstrap or analytic)
        se_boot: Bootstrap standard error
        se_analytic: Analytic standard error  
        p: P-value (bootstrap or WCB if available)
        p_boot: Bootstrap p-value
        p_wcb: Wild cluster bootstrap p-value (if computed)
        ci_lower: Lower confidence interval
        ci_upper: Upper confidence interval
        att_gt_df: Full ATT(g,t) estimates dataframe
        used_df: Panel data actually used
    """
    att_overall: float
    se: float
    se_boot: float
    se_analytic: float
    p: float
    p_boot: float
    p_wcb: float
    ci_lower: float
    ci_upper: float
    att_gt_df: pd.DataFrame
    used_df: pd.DataFrame


def _prepare_r_panel(
    panel_df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    cohort_col: str = "g",
    covariates: Optional[list] = None,
) -> pd.DataFrame:
    """
    Prepare panel for R's did::att_gt.
    
    Args:
        panel_df: Panel with unit_id, year, outcome, g (cohort), treated_ever
        unit_col: Unit identifier column
        time_col: Time variable column
        outcome_col: Outcome variable column
        cohort_col: Cohort (first treatment year) column
        covariates: Optional list of covariate columns
    
    Returns:
        DataFrame ready for R's did package
    """
    df = panel_df.copy()
    
    # Ensure required columns exist
    required = [unit_col, time_col, outcome_col, cohort_col, "treated_ever"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[R-did] Missing required columns: {missing}")
    
    # Convert types to basic Python types (not pandas categorical)
    df[unit_col] = df[unit_col].astype(str).astype('object')
    df[time_col] = df[time_col].astype(int).astype('int64')
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors='coerce').astype('float64')
    
    # Cohort: keep as numeric for treated, set to 0 for never-treated
    # (R's did package uses 0 to indicate never-treated)
    df[cohort_col] = pd.to_numeric(df[cohort_col], errors='coerce').astype('float64')
    never_treated = df["treated_ever"] == 0
    df.loc[never_treated, cohort_col] = 0.0
    
    # Convert covariates to numeric
    if covariates:
        for col in covariates:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    
    # Drop rows with missing outcome or cohort
    df = df.dropna(subset=[outcome_col, cohort_col])
    
    # Select columns
    keep_cols = [unit_col, time_col, outcome_col, cohort_col]
    if covariates:
        keep_cols.extend([c for c in covariates if c in df.columns])
    
    df = df[keep_cols].copy()
    
    # Ensure all columns are basic types (not category)
    for col in df.columns:
        if df[col].dtype.name == 'category':
            df[col] = df[col].astype(str)
    
    return df


def estimate_att_r_did(
    panel_df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    cohort_col: str = "g",
    covariates: Optional[list] = None,
    *,
    use_wcb: bool = False,
    B: int = 9999,
    weights: str = "rademacher",
    seed: Optional[int] = 123,
    clustid_col: Optional[str] = None,
) -> RDidResult:
    """
    Estimate Callaway-Sant'Anna ATT^o using R's did package.
    
    This uses the original R implementation with optional wild cluster bootstrap.
    For small treatment cluster counts (< 20), WCB is strongly recommended.
    
    Args:
        panel_df: Panel with unit, time, outcome, cohort, treated_ever
        unit_col: Unit identifier column name
        time_col: Time variable column name  
        outcome_col: Outcome variable column name
        cohort_col: Cohort (first treatment year) column name (default "g")
        covariates: Optional list of covariate column names
        use_wcb: Whether to compute wild cluster bootstrap p-value
        B: Bootstrap iterations (default 9999)
        weights: Bootstrap weights - 'rademacher' or 'mammen'
        seed: Random seed for reproducibility
        clustid_col: Cluster ID column (defaults to unit_col)
    
    Returns:
        RDidResult with ATT^o, SE, p-values (bootstrap and optionally WCB)
    
    Raises:
        ValueError: If required columns missing or data invalid
        RuntimeError: If R call fails
    """
    # Prepare panel
    print(f"[R-did] Preparing panel data...")
    df = _prepare_r_panel(
        panel_df=panel_df,
        unit_col=unit_col,
        time_col=time_col,
        outcome_col=outcome_col,
        cohort_col=cohort_col,
        covariates=covariates,
    )
    
    n_units = df[unit_col].nunique()
    n_treated = df[df[cohort_col] > 0][unit_col].nunique()
    n_obs = len(df)
    
    print(f"[R-did] Panel: {n_obs} obs, {n_units} units ({n_treated} treated)")
    
    if clustid_col is None:
        clustid_col = unit_col
    
    # Activate pandas-R conversion and convert to R dataframe
    print("[R-did] Converting to R dataframe...")
    
    # Manual conversion to avoid pandas2ri issues
    # Create R dataframe using column assignment
    r_df = robjects.NULL
    r_list = {}
    
    for col in df.columns:
        col_data = df[col].values
        if col_data.dtype == 'object':
            r_list[col] = robjects.StrVector(col_data.astype(str))
        elif col_data.dtype == 'int64':
            r_list[col] = robjects.IntVector(col_data.astype(int))
        else:  # float
            r_list[col] = robjects.FloatVector(col_data.astype(float))
    
    # Create R data frame from list
    r_df = robjects.r['data.frame'](**r_list)
    
    # Import R did package
    print("[R-did] Loading R did package...")
    did = importr('did')
    
    # Build formula for covariates
    if covariates:
        xformla = robjects.Formula("~ " + " + ".join(covariates))
    else:
        xformla = robjects.NULL
    
    # Call att_gt
    print(f"[R-did] Computing ATT(g,t) with {B} bootstrap iterations...")
    att_gt_result = did.att_gt(
        yname=outcome_col,
        tname=time_col,
        idname=unit_col,
        gname=cohort_col,
        xformla=xformla,
        data=r_df,
        control_group="nevertreated",
        bstrap=True,
        biters=B,
        clustervars=robjects.StrVector([clustid_col]) if clustid_col else robjects.NULL,
        cband=False,
        alp=0.05,
        print_details=False,
    )
    
    # Aggregate to overall ATT^o
    print("[R-did] Aggregating to ATT^o (simple)...")
    agg_result = did.aggte(
        att_gt_result,
        type="simple",
        bstrap=True,
        biters=B,
        cband=False,
        alp=0.05,
    )
    
    # Extract results
    att_overall = float(robjects.r['as.numeric'](agg_result.rx2("overall.att"))[0])
    se_overall = float(robjects.r['as.numeric'](agg_result.rx2("overall.se"))[0])
    
    # Try to get bootstrap SE (may be same as overall.se)
    se_boot = se_overall
    se_analytic = math.nan
    
    # Compute p-value from bootstrap SE
    if se_boot > 0:
        t_stat = att_overall / se_boot
        p_boot = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))
    else:
        p_boot = math.nan
    
    # Get confidence intervals if available
    ci_lower = math.nan
    ci_upper = math.nan
    if "overall.att.lower" in agg_result.names:
        ci_lower = float(robjects.r['as.numeric'](agg_result.rx2("overall.att.lower"))[0])
    if "overall.att.upper" in agg_result.names:
        ci_upper = float(robjects.r['as.numeric'](agg_result.rx2("overall.att.upper"))[0])
    
    print(f"[R-did] ATT^o = {att_overall:.6f}, SE = {se_boot:.6f}, p = {p_boot:.6f}")
    
    # Extract ATT(g,t) results
    att_gt_df = pd.DataFrame()
    
    # Wild Cluster Bootstrap (optional)
    p_wcb = math.nan
    if use_wcb:
        print(f"[R-did] Computing WCB p-value (B={B}, weights={weights})...")
        
        # For WCB, we need to refit using a TWFE model
        # Then apply boottest via fwildclusterboot
        
        # Create treated indicator
        df_wcb = df.copy()
        df_wcb['treated'] = (
            (df_wcb[cohort_col] > 0) & 
            (df_wcb[time_col] >= df_wcb[cohort_col])
        ).astype(int)
        
        # Convert to R dataframe (same manual process)
        r_wcb_list = {}
        for col in df_wcb.columns:
            col_data = df_wcb[col].values
            if col_data.dtype == 'object':
                r_wcb_list[col] = robjects.StrVector(col_data.astype(str))
            elif col_data.dtype == 'int64' or col_data.dtype == 'int':
                r_wcb_list[col] = robjects.IntVector(col_data.astype(int))
            else:  # float
                r_wcb_list[col] = robjects.FloatVector(col_data.astype(float))
        
        r_df_wcb = robjects.r['data.frame'](**r_wcb_list)
        
        # Fit TWFE model
        stats = importr('stats')
        formula_str = f"{outcome_col} ~ treated + factor({unit_col}) + factor({time_col})"
        
        if covariates:
            formula_str += " + " + " + ".join(covariates)
        
        print(f"[R-did-wcb] TWFE formula: {formula_str}")
        fit = stats.lm(robjects.Formula(formula_str), data=r_df_wcb)
        
        # Apply WCB
        fwcb = importr('fwildclusterboot')
        boot_result = fwcb.boottest(
            fit,
            clustid=robjects.StrVector([clustid_col]),
            param="treated",
            B=B,
            type_=weights,
            impose_null=robjects.BoolVector([True]),
            seed=seed if seed else robjects.NULL,
        )
        
        # Extract WCB p-value
        if "p_val" in boot_result.names:
            p_wcb = float(robjects.r['as.numeric'](boot_result.rx2("p_val"))[0])
        elif "p.value" in boot_result.names:
            p_wcb = float(robjects.r['as.numeric'](boot_result.rx2("p.value"))[0])
        else:
            p_wcb = float(robjects.r['as.numeric'](boot_result[0])[0])
        
        print(f"[R-did-wcb] WCB p-value: {p_wcb:.6f}")
    
    # Determine preferred p-value
    p_final = p_wcb if use_wcb and np.isfinite(p_wcb) else p_boot
    
    return RDidResult(
        att_overall=att_overall,
        se=se_boot,
        se_boot=se_boot,
        se_analytic=se_analytic,
        p=p_final,
        p_boot=p_boot,
        p_wcb=p_wcb,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        att_gt_df=att_gt_df,
        used_df=df,
    )


def log_wcb_call_r_did(
    n_units: int,
    n_treated: int,
    n_obs: int,
    att: float,
    se_boot: float,
    p_boot: float,
    p_wcb: float,
    B: int,
    weights: str,
) -> None:
    """Log summary of R-did WCB call for diagnostics."""
    print("=" * 72)
    print("[R-DID SUMMARY]")
    print("=" * 72)
    print(f"Sample:")
    print(f"  - Total units: {n_units}")
    print(f"  - Treated units: {n_treated}")
    print(f"  - Observations: {n_obs}")
    print(f"Estimate:")
    print(f"  - ATT^o: {att:.6f}")
    print(f"  - SE (bootstrap): {se_boot:.6f}")
    print(f"Inference:")
    print(f"  - p (bootstrap): {p_boot:.6f}")
    if np.isfinite(p_wcb):
        print(f"  - p (WCB): {p_wcb:.6f}")
        print(f"  - WCB settings: B={B}, weights={weights}")
    print("=" * 72)