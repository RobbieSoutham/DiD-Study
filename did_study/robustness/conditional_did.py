# did_study/robustness/conditional_did.py
"""
Conditional parallel trends pre-test using the `did` package in R.

This module provides a wrapper around `conditional_did_pretest()` from the
Callaway and Sant'Anna (2021) `did` package to test whether the conditional
parallel trends assumption holds in all pre-treatment periods for all groups.

Reference:
    Callaway, Brantly and Sant'Anna, Pedro H. C. (2021). "Difference-in-Differences 
    with Multiple Time Periods." Journal of Econometrics, Vol. 225, No. 2, pp. 200-230.
    https://doi.org/10.1016/j.jeconom.2020.12.001
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from did_study.robustness.r_interface import set_r_seeds


@dataclass
class ConditionalDiDPretestResult:
    """
    Container for conditional parallel trends pre-test results.
    
    The test returns two statistics:
    - Cramer von Mises (CvM): Integrated squared moment statistic
    - Kolmogorov-Smirnov (KS): Supremum absolute moment statistic
    
    Both statistics test H0: Conditional parallel trends holds in all 
    pre-treatment periods for all groups.
    """
    # Cramer von Mises test
    CvM: float  # Test statistic
    CvM_pval: float  # P-value (bootstrapped)
    CvM_cval: float  # Critical value at specified alpha
    
    # Kolmogorov-Smirnov test
    KS: float  # Test statistic
    KS_pval: float  # P-value (bootstrapped)
    KS_cval: float  # Critical value at specified alpha
    
    # Test configuration
    alp: float  # Significance level used
    biters: int  # Number of bootstrap iterations
    
    # Additional metadata
    control_group: str  # "nevertreated" or "notyettreated"
    est_method: str  # Estimation method ("ipw", "dr", or "reg")
    xformla: Optional[str]  # Covariate formula
    
    def to_dict(self) -> Dict[str, Any]:
        """Export results as dictionary."""
        return {
            "CvM_statistic": self.CvM,
            "CvM_pvalue": self.CvM_pval,
            "CvM_critical_value": self.CvM_cval,
            "KS_statistic": self.KS,
            "KS_pvalue": self.KS_pval,
            "KS_critical_value": self.KS_cval,
            "significance_level": self.alp,
            "bootstrap_iterations": self.biters,
            "control_group": self.control_group,
            "estimation_method": self.est_method,
            "covariate_formula": self.xformla,
        }
    
    def passed(self, test: str = "both", alpha: Optional[float] = None) -> bool:
        """
        Check if the conditional parallel trends assumption is satisfied.
        
        Parameters
        ----------
        test : str
            Which test to use: "CvM", "KS", or "both" (default).
            "both" requires both tests to pass.
        alpha : float, optional
            Significance level. If None, uses self.alp.
        
        Returns
        -------
        bool
            True if H0 (conditional PT holds) is NOT rejected.
        """
        alpha = alpha or self.alp
        
        if test == "CvM":
            return self.CvM_pval > alpha
        elif test == "KS":
            return self.KS_pval > alpha
        elif test == "both":
            return (self.CvM_pval > alpha) and (self.KS_pval > alpha)
        else:
            raise ValueError(f"Invalid test type: {test}. Must be 'CvM', 'KS', or 'both'.")


def conditional_did_pretest(
    data: pd.DataFrame,
    yname: str,
    tname: str,
    gname: str,
    idname: Optional[str] = None,
    xformla: Optional[str] = None,
    panel: bool = True,
    allow_unbalanced_panel: bool = False,
    control_group: str = "nevertreated",
    weightsname: Optional[str] = None,
    alp: float = 0.05,
    bstrap: bool = True,
    cband: bool = True,
    biters: int = 1000,
    clustervars: Optional[Sequence[str]] = None,
    est_method: str = "ipw",
    print_details: bool = False,
    pl: bool = False,
    cores: int = 1,
    seed: Optional[int] = 123,
) -> ConditionalDiDPretestResult:
    """
    Test the conditional parallel trends assumption using the did package.
    
    This function implements an integrated moments test for whether the conditional
    parallel trends assumption holds in all pre-treatment time periods for all groups.
    
    The test returns two statistics:
    - Cramer von Mises (CvM): More powerful against smooth violations
    - Kolmogorov-Smirnov (KS): More powerful against sharp violations
    
    If p-value > alpha (typically 0.05), you fail to reject H0 and conclude that
    conditional parallel trends is plausible.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format (one row per unit-time observation).
        
    yname : str
        Name of the outcome variable column.
        
    tname : str
        Name of the time period column.
        
    gname : str
        Name of the column indicating first treatment period for each unit.
        Should be a positive integer for treated units and 0 for never-treated units.
        
    idname : str, optional
        Name of the unit ID column. Required if panel=True.
        
    xformla : str, optional
        R formula string for covariates to condition on (e.g., "~lpop+lemployment").
        Default is None, which is equivalent to "~1" (no covariates).
        This is the KEY parameter for testing conditional parallel trends.
        
    panel : bool
        Whether data is panel (True) or repeated cross-sections (False).
        
    allow_unbalanced_panel : bool
        Whether to allow unbalanced panels. If False, drops units not observed
        in all periods.
        
    control_group : str
        Which units form the control group:
        - "nevertreated": Units that never receive treatment (default)
        - "notyettreated": Units not yet treated in that period
        
    weightsname : str, optional
        Name of sampling weights column.
        
    alp : float
        Significance level for the test (default 0.05).
        
    bstrap : bool
        Whether to use multiplier bootstrap for standard errors.
        
    cband : bool
        Whether to compute uniform confidence bands.
        
    biters : int
        Number of bootstrap iterations (default 1000).
        
    clustervars : Sequence[str], optional
        Variables to cluster on (max 2, one should be idname).
        
    est_method : str
        Estimation method for group-time effects:
        - "ipw": Inverse probability weighting (default for pretest)
        - "dr": Doubly robust
        - "reg": Regression
        
    print_details : bool
        Whether to print progress/details.
        
    pl : bool
        Whether to use parallel processing.
        
    cores : int
        Number of cores for parallel processing.
        
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    ConditionalDiDPretestResult
        Object containing test statistics, p-values, and critical values.
    
    Raises
    ------
    ValueError
        If required columns are missing or data format is invalid.
    RuntimeError
        If the R call fails.
    
    Examples
    --------
    >>> # Test unconditional parallel trends
    >>> result = conditional_did_pretest(
    ...     data=df,
    ...     yname="outcome",
    ...     tname="year",
    ...     idname="unit_id",
    ...     gname="first_treat",
    ...     xformla=None,  # No covariates
    ... )
    >>> print(f"CvM p-value: {result.CvM_pval:.4f}")
    >>> print(f"Test passed: {result.passed()}")
    
    >>> # Test conditional parallel trends (conditioning on dose)
    >>> result = conditional_did_pretest(
    ...     data=df,
    ...     yname="outcome",
    ...     tname="year",
    ...     idname="unit_id",
    ...     gname="first_treat",
    ...     xformla="~dose",  # Condition on dose
    ... )
    >>> if result.passed():
    ...     print("Conditional PT assumption plausible")
    ... else:
    ...     print("Conditional PT assumption violated")
    """
    
    # Validate inputs
    required_cols = [yname, tname, gname]
    if panel and idname:
        required_cols.append(idname)
    if weightsname:
        required_cols.append(weightsname)
    
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate control_group
    if control_group not in ["nevertreated", "notyettreated"]:
        raise ValueError(
            f"control_group must be 'nevertreated' or 'notyettreated', got '{control_group}'"
        )
    
    # Import did package
    try:
        did = importr("did")
    except Exception as e:
        raise RuntimeError(
            f"Failed to import R 'did' package. Make sure it's installed: "
            f"install.packages('did'). Error: {e}"
        ) from e
    
    # Set seed
    set_r_seeds(seed)
    
    # Convert data to R
    with (ro.default_converter + pandas2ri.converter).context():
        R_data = ro.conversion.get_conversion().py2rpy(data)
    
    # Prepare R arguments
    kwargs: Dict[str, Any] = {
        "yname": yname,
        "tname": tname,
        "gname": gname,
        "data": R_data,
        "panel": panel,
        "allow_unbalanced_panel": allow_unbalanced_panel,
        "control_group": control_group,
        "alp": alp,
        "bstrap": bstrap,
        "cband": cband,
        "biters": int(biters),
        "est_method": est_method,
        "print_details": print_details,
        "pl": pl,
        "cores": int(cores),
    }
    
    # Add optional parameters
    if idname:
        kwargs["idname"] = idname
    
    if xformla is not None:
        # Convert Python formula to R formula
        if not xformla.startswith("~"):
            xformla = f"~{xformla}"
        kwargs["xformla"] = ro.Formula(xformla)
    
    if weightsname:
        kwargs["weightsname"] = weightsname
    
    if clustervars:
        kwargs["clustervars"] = ro.StrVector(list(clustervars))
    
    # Print call information
    print("=" * 72)
    print("[CONDITIONAL DID PRETEST] -> did::conditional_did_pretest")
    print("=" * 72)
    print("Parameters:")
    print(f"  - outcome: {yname}")
    print(f"  - time: {tname}")
    print(f"  - id: {idname}")
    print(f"  - group: {gname}")
    print(f"  - covariates: {xformla or 'None (unconditional test)'}")
    print(f"  - control_group: {control_group}")
    print(f"  - panel: {panel}")
    print(f"  - est_method: {est_method}")
    print(f"  - alpha: {alp}")
    print(f"  - bootstrap iterations: {biters}")
    print(f"  - seed: {seed}")
    print()
    
    # Call R function
    try:
        R_result = did.conditional_did_pretest(**kwargs)
    except Exception as e:
        raise RuntimeError(
            f"conditional_did_pretest R call failed: {e}"
        ) from e
    
    # Extract results from MP.TEST object
    try:
        # R object contains: CvM, CvMb, CvMcval, CvMpval, KS, KSb, KScval, KSpval
        
        # Extract scalar values
        CvM = float(np.array(R_result.rx2("CvM"))[0])
        CvM_pval = float(np.array(R_result.rx2("CvMpval"))[0])
        CvM_cval = float(np.array(R_result.rx2("CvMcval"))[0])
        
        KS = float(np.array(R_result.rx2("KS"))[0])
        KS_pval = float(np.array(R_result.rx2("KSpval"))[0])
        KS_cval = float(np.array(R_result.rx2("KScval"))[0])
        
        # Print results
        print(f"Conditional DiD Pre-Test Results:")
        print(f"  Cramer von Mises:")
        print(f"    - Statistic: {CvM:.4f}")
        print(f"    - P-value: {CvM_pval:.4f}")
        print(f"    - Critical value ({alp}): {CvM_cval:.4f}")
        print(f"  Kolmogorov-Smirnov:")
        print(f"    - Statistic: {KS:.4f}")
        print(f"    - P-value: {KS_pval:.4f}")
        print(f"    - Critical value ({alp}): {KS_cval:.4f}")
        print()
        
        # Interpretation
        passed = (CvM_pval > alp) and (KS_pval > alp)
        if passed:
            print(f"✓ PASS: Conditional parallel trends assumption NOT rejected at {alp} level")
            print(f"  Both p-values > {alp}: CvM={CvM_pval:.4f}, KS={KS_pval:.4f}")
        else:
            print(f"✗ FAIL: Conditional parallel trends assumption REJECTED at {alp} level")
            if CvM_pval <= alp:
                print(f"  CvM test rejected (p={CvM_pval:.4f} ≤ {alp})")
            if KS_pval <= alp:
                print(f"  KS test rejected (p={KS_pval:.4f} ≤ {alp})")
        print()
        
        return ConditionalDiDPretestResult(
            CvM=CvM,
            CvM_pval=CvM_pval,
            CvM_cval=CvM_cval,
            KS=KS,
            KS_pval=KS_pval,
            KS_cval=KS_cval,
            alp=alp,
            biters=biters,
            control_group=control_group,
            est_method=est_method,
            xformla=xformla,
        )
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse conditional_did_pretest output: {e}"
        ) from e


def test_dose_conditional_pt(
    data: pd.DataFrame,
    yname: str,
    tname: str,
    gname: str,
    idname: str,
    dose_col: str,
    **kwargs,
) -> Dict[str, ConditionalDiDPretestResult]:
    """
    Convenience function to test both unconditional and dose-conditional PT.
    
    This runs the pretest twice:
    1. Without conditioning on dose (unconditional/standard PT)
    2. Conditioning on dose (conditional PT)
    
    Comparing the two helps diagnose whether dose heterogeneity violates PT.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    yname, tname, gname, idname : str
        Column names for outcome, time, group, and unit ID.
    dose_col : str
        Name of the dose/treatment intensity column.
    **kwargs
        Additional arguments passed to conditional_did_pretest.
    
    Returns
    -------
    dict
        Dictionary with keys "unconditional" and "conditional", each containing
        a ConditionalDiDPretestResult object.
    
    Examples
    --------
    >>> results = test_dose_conditional_pt(
    ...     data=df,
    ...     yname="outcome",
    ...     tname="year",
    ...     gname="first_treat",
    ...     idname="unit_id",
    ...     dose_col="dose",
    ... )
    >>> 
    >>> print("\\nUnconditional PT:")
    >>> print(f"  Passed: {results['unconditional'].passed()}")
    >>> print(f"  CvM p-value: {results['unconditional'].CvM_pval:.4f}")
    >>> 
    >>> print("\\nConditional PT (controlling for dose):")
    >>> print(f"  Passed: {results['conditional'].passed()}")
    >>> print(f"  CvM p-value: {results['conditional'].CvM_pval:.4f}")
    """
    
    print("=" * 72)
    print("DOSE HETEROGENEITY DIAGNOSTIC")
    print("=" * 72)
    print()
    
    # Test 1: Unconditional PT (standard assumption)
    print("TEST 1: UNCONDITIONAL PARALLEL TRENDS")
    print("-" * 72)
    unconditional = conditional_did_pretest(
        data=data,
        yname=yname,
        tname=tname,
        gname=gname,
        idname=idname,
        xformla=None,  # No covariates
        **kwargs,
    )
    
    print()
    
    # Test 2: Conditional PT (controlling for dose)
    print("TEST 2: CONDITIONAL PARALLEL TRENDS (conditioning on dose)")
    print("-" * 72)
    conditional = conditional_did_pretest(
        data=data,
        yname=yname,
        tname=tname,
        gname=gname,
        idname=idname,
        xformla=f"~{dose_col}",  # Condition on dose
        **kwargs,
    )
    
    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    
    unc_pass = unconditional.passed()
    cond_pass = conditional.passed()
    
    if unc_pass and cond_pass:
        print("✓ BOTH TESTS PASS")
        print("  Standard parallel trends holds (with or without dose conditioning)")
        print("  → Your binarised long-difference approach is valid for ATT^o")
    elif not unc_pass and cond_pass:
        print("⚠ UNCONDITIONAL FAILS, CONDITIONAL PASSES")
        print("  Dose heterogeneity causes PT violation in raw trends")
        print("  But PT holds after conditioning on dose")
        print("  → Need to include dose as covariate in your analysis")
    elif unc_pass and not cond_pass:
        print("⚠ UNCONDITIONAL PASSES, CONDITIONAL FAILS (unusual)")
        print("  This is rare and suggests model misspecification")
        print("  → Check covariate specification and functional form")
    else:
        print("✗ BOTH TESTS FAIL")
        print("  Parallel trends violated even after dose conditioning")
        print("  → Your identifying assumption does not hold")
        print("  → Consider alternative identification strategies")
    
    print()
    
    return {
        "unconditional": unconditional,
        "conditional": conditional,
    }