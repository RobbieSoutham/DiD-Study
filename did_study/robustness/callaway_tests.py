# did_study/robustness/callaway_tests.py
"""
Comprehensive testing suite for Callaway et al. (2024) continuous DiD.

This module implements the full diagnostic and robustness testing workflow
for continuous/dose-varying treatment difference-in-differences designs.

References:
    Callaway, Brantly, Goodman-Bacon, Andrew, and Sant'Anna, Pedro H. C. (2024).
    "Difference-in-Differences with a Continuous Treatment."
    NBER Working Paper No. 32117.
    
    Callaway, Brantly and Sant'Anna, Pedro H. C. (2021).
    "Difference-in-Differences with Multiple Time Periods."
    Journal of Econometrics, Vol. 225, No. 2, pp. 200-230.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from did_study.robustness.conditional_did import (
    conditional_did_pretest,
    ConditionalDiDPretestResult,
)
from did_study.robustness.honest_did import (
    honest_did_bounds,
    HonestDiDResult,
)


@dataclass
class CallawayTestSuite:
    """
    Container for complete Callaway et al. 2024 testing results.
    
    Includes:
    - Unconditional parallel trends test (standard PT)
    - Conditional parallel trends test (dose-adjusted PT)
    - Per-bin diagnostics (if conditional test fails)
    - HonestDiD sensitivity bounds (optional robustness)
    """
    # Core PT tests
    unconditional_pt: ConditionalDiDPretestResult
    conditional_pt: ConditionalDiDPretestResult
    
    # Per-bin diagnostics (only if conditional fails)
    bin_pt_tests: Optional[Dict[str, ConditionalDiDPretestResult]] = None
    
    # HonestDiD bounds (optional)
    honest_did: Optional[HonestDiDResult] = None
    
    # Metadata
    dose_col: str = "dose"
    n_bins: Optional[int] = None
    
    def passed_validation(self, require_both: bool = False) -> bool:
        """
        Check if identifying assumptions are satisfied.
        
        Parameters
        ----------
        require_both : bool
            If True, require both unconditional and conditional to pass.
            If False, only require conditional to pass (recommended).
        
        Returns
        -------
        bool
            True if validation passed.
        """
        if require_both:
            return self.unconditional_pt.passed() and self.conditional_pt.passed()
        else:
            return self.conditional_pt.passed()
    
    def diagnosis(self) -> str:
        """
        Return a human-readable diagnosis of test results.
        
        Returns
        -------
        str
            Diagnostic message.
        """
        unc_pass = self.unconditional_pt.passed()
        cond_pass = self.conditional_pt.passed()
        
        if unc_pass and cond_pass:
            return (
                "PASS: Both unconditional and conditional PT hold. "
                "Your binning strategy is valid. Proceed with estimation."
            )
        elif not unc_pass and cond_pass:
            return (
                "CONDITIONAL PASS: Unconditional PT fails but conditional PT holds. "
                "Dose heterogeneity is present but accounted for. "
                "Include dose as covariate in your model."
            )
        elif unc_pass and not cond_pass:
            return (
                "WARNING: Unconditional PT passes but conditional PT fails. "
                "This is unusual and suggests model misspecification. "
                "Check functional form of dose relationship."
            )
        else:
            return (
                "FAIL: Both unconditional and conditional PT fail. "
                "Parallel trends assumption fundamentally violated. "
                "Do not proceed with standard DiD estimation."
            )
    
    def summary(self) -> str:
        """Generate formatted summary of all test results."""
        lines = []
        lines.append("=" * 70)
        lines.append("CALLAWAY ET AL. 2024 TESTING SUITE RESULTS")
        lines.append("=" * 70)
        lines.append("")
        
        # Unconditional PT
        lines.append("1. UNCONDITIONAL PARALLEL TRENDS TEST")
        lines.append("-" * 70)
        lines.append(f"   Cramer von Mises: stat={self.unconditional_pt.CvM:.4f}, "
                    f"p={self.unconditional_pt.CvM_pval:.4f}")
        lines.append(f"   Kolmogorov-Smirnov: stat={self.unconditional_pt.KS:.4f}, "
                    f"p={self.unconditional_pt.KS_pval:.4f}")
        lines.append(f"   Result: {'PASS' if self.unconditional_pt.passed() else 'FAIL'}")
        lines.append("")
        
        # Conditional PT
        lines.append(f"2. CONDITIONAL PARALLEL TRENDS TEST (on {self.dose_col})")
        lines.append("-" * 70)
        lines.append(f"   Cramer von Mises: stat={self.conditional_pt.CvM:.4f}, "
                    f"p={self.conditional_pt.CvM_pval:.4f}")
        lines.append(f"   Kolmogorov-Smirnov: stat={self.conditional_pt.KS:.4f}, "
                    f"p={self.conditional_pt.KS_pval:.4f}")
        lines.append(f"   Result: {'PASS' if self.conditional_pt.passed() else 'FAIL'}")
        lines.append("")
        
        # Per-bin diagnostics
        if self.bin_pt_tests:
            lines.append("3. PER-BIN DIAGNOSTICS")
            lines.append("-" * 70)
            for bin_label, result in self.bin_pt_tests.items():
                status = 'PASS' if result.passed() else 'FAIL'
                lines.append(f"   {bin_label}: {status} "
                           f"(CvM p={result.CvM_pval:.4f}, KS p={result.KS_pval:.4f})")
            lines.append("")
        
        # HonestDiD
        if self.honest_did:
            lines.append("4. HONESTDID SENSITIVITY BOUNDS")
            lines.append("-" * 70)
            lines.append(f"   Method: {self.honest_did.method} ({self.honest_did.delta_label})")
            lines.append(f"   M-grid: {self.honest_did.M}")
            lines.append(f"   Lower bounds: {self.honest_did.lb}")
            lines.append(f"   Upper bounds: {self.honest_did.ub}")
            lines.append("")
        
        # Overall diagnosis
        lines.append("=" * 70)
        lines.append("OVERALL DIAGNOSIS")
        lines.append("=" * 70)
        lines.append(self.diagnosis())
        lines.append("=" * 70)
        
        return "\n".join(lines)


def run_callaway_tests(
    data: pd.DataFrame,
    yname: str,
    tname: str,
    gname: str,
    idname: str,
    dose_col: str,
    # PT test parameters
    biters: int = 9999,
    alp: float = 0.05,
    control_group: str = "nevertreated",
    est_method: str = "ipw",
    seed: Optional[int] = 123,
    # Per-bin diagnostics
    run_bin_diagnostics: bool = True,
    n_bins: int = 4,
    bin_labels: Optional[List[str]] = None,
    # HonestDiD parameters
    run_honest_did: bool = False,
    event_study_betas: Optional[np.ndarray] = None,
    event_study_Sigma: Optional[np.ndarray] = None,
    numPrePeriods: Optional[int] = None,
    numPostPeriods: Optional[int] = None,
    Mmax: float = 2.0,
    honest_grid_points: int = 10,
    # Output control
    verbose: bool = True,
) -> CallawayTestSuite:
    """
    Run complete Callaway et al. 2024 testing suite for continuous DiD.
    
    This is the main entry point for comprehensive diagnostic testing of
    parallel trends assumptions in continuous/dose-varying treatment designs.
    
    Testing hierarchy:
    1. Unconditional PT test (baseline check)
    2. Conditional PT test on dose (key test for dose heterogeneity)
    3. Per-bin PT diagnostics (only if conditional test fails)
    4. HonestDiD sensitivity bounds (optional robustness check)
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    yname : str
        Outcome variable column name.
    tname : str
        Time period column name.
    gname : str
        First treatment period column name (0 for never-treated).
    idname : str
        Unit ID column name.
    dose_col : str
        Dose/treatment intensity column name.
    biters : int
        Number of bootstrap iterations for PT tests (default 9999).
    alp : float
        Significance level (default 0.05).
    control_group : str
        "nevertreated" or "notyettreated" (default "nevertreated").
    est_method : str
        Estimation method: "ipw", "dr", or "reg" (default "ipw").
    seed : int, optional
        Random seed for reproducibility.
    run_bin_diagnostics : bool
        Whether to run per-bin PT tests if conditional test fails (default True).
    n_bins : int
        Number of dose bins for diagnostics (default 4).
    bin_labels : List[str], optional
        Custom bin labels. If None, uses ["Q1", "Q2", ..., "Qn"].
    run_honest_did : bool
        Whether to run HonestDiD sensitivity analysis (default False).
    event_study_betas : np.ndarray, optional
        Event-study coefficients for HonestDiD (required if run_honest_did=True).
    event_study_Sigma : np.ndarray, optional
        Covariance matrix for HonestDiD (required if run_honest_did=True).
    numPrePeriods : int, optional
        Number of pre-periods for HonestDiD.
    numPostPeriods : int, optional
        Number of post-periods for HonestDiD.
    Mmax : float
        Maximum relative magnitude for HonestDiD (default 2.0).
    honest_grid_points : int
        Number of M-grid points for HonestDiD (default 10).
    verbose : bool
        Print progress messages (default True).
    
    Returns
    -------
    CallawayTestSuite
        Complete test results object.
    
    Examples
    --------
    >>> # Basic usage: just PT tests
    >>> results = run_callaway_tests(
    ...     data=panel_df,
    ...     yname="outcome",
    ...     tname="year",
    ...     gname="first_treat",
    ...     idname="unit_id",
    ...     dose_col="dose",
    ... )
    >>> print(results.summary())
    >>> if results.passed_validation():
    ...     print("Proceed with estimation")
    
    >>> # Full suite with HonestDiD
    >>> results = run_callaway_tests(
    ...     data=panel_df,
    ...     yname="outcome",
    ...     tname="year",
    ...     gname="first_treat",
    ...     idname="unit_id",
    ...     dose_col="dose",
    ...     run_honest_did=True,
    ...     event_study_betas=betas,
    ...     event_study_Sigma=Sigma,
    ...     numPrePeriods=4,
    ...     numPostPeriods=8,
    ... )
    >>> print(results.summary())
    """
    
    if verbose:
        print("=" * 70)
        print("CALLAWAY ET AL. 2024 TESTING SUITE")
        print("=" * 70)
        print(f"Data: {len(data)} observations")
        print(f"Outcome: {yname}")
        print(f"Dose variable: {dose_col}")
        print(f"Bootstrap iterations: {biters}")
        print(f"Significance level: {alp}")
        print()
    
    # =========================================================================
    # TIER 1: UNCONDITIONAL PARALLEL TRENDS TEST
    # =========================================================================
    if verbose:
        print("TIER 1: UNCONDITIONAL PARALLEL TRENDS TEST")
        print("-" * 70)
    
    result_unc = conditional_did_pretest(
        data=data,
        yname=yname,
        tname=tname,
        gname=gname,
        idname=idname,
        xformla=None,  # No covariates = unconditional
        control_group=control_group,
        alp=alp,
        bstrap=True,
        cband=True,
        biters=biters,
        est_method=est_method,
        print_details=False,
        seed=seed,
    )
    
    if verbose:
        status = "PASS" if result_unc.passed() else "FAIL"
        print(f"Result: {status}")
        print(f"  CvM: stat={result_unc.CvM:.4f}, p={result_unc.CvM_pval:.4f}")
        print(f"  KS:  stat={result_unc.KS:.4f}, p={result_unc.KS_pval:.4f}")
        print()
    
    # =========================================================================
    # TIER 2: CONDITIONAL PARALLEL TRENDS TEST (KEY TEST)
    # =========================================================================
    if verbose:
        print("TIER 2: CONDITIONAL PARALLEL TRENDS TEST")
        print("-" * 70)
        print(f"Conditioning on: {dose_col}")
    
    result_cond = conditional_did_pretest(
        data=data,
        yname=yname,
        tname=tname,
        gname=gname,
        idname=idname,
        xformla=f"~{dose_col}",  # Condition on dose
        control_group=control_group,
        alp=alp,
        bstrap=True,
        cband=True,
        biters=biters,
        est_method=est_method,
        print_details=False,
        seed=seed,
    )
    
    if verbose:
        status = "PASS" if result_cond.passed() else "FAIL"
        print(f"Result: {status}")
        print(f"  CvM: stat={result_cond.CvM:.4f}, p={result_cond.CvM_pval:.4f}")
        print(f"  KS:  stat={result_cond.KS:.4f}, p={result_cond.KS_pval:.4f}")
        print()
    
    # =========================================================================
    # TIER 3: PER-BIN DIAGNOSTICS (only if conditional test fails)
    # =========================================================================
    bin_results = None
    
    if run_bin_diagnostics and not result_cond.passed():
        if verbose:
            print("TIER 3: PER-BIN DIAGNOSTICS")
            print("-" * 70)
            print("Conditional PT test failed. Running diagnostics per dose bin...")
            print()
        
        # Create dose bins
        if bin_labels is None:
            bin_labels = [f"Q{i+1}" for i in range(n_bins)]
        
        try:
            data_copy = data.copy()
            data_copy['_dose_bin'] = pd.qcut(
                data_copy[dose_col],
                q=n_bins,
                labels=bin_labels,
                duplicates='drop'
            )
        except ValueError as e:
            if verbose:
                print(f"Warning: Could not create {n_bins} bins (duplicate edges). "
                      f"Using fewer bins.")
            # Try with fewer bins
            for alt_bins in range(n_bins-1, 1, -1):
                try:
                    alt_labels = [f"Q{i+1}" for i in range(alt_bins)]
                    data_copy['_dose_bin'] = pd.qcut(
                        data_copy[dose_col],
                        q=alt_bins,
                        labels=alt_labels,
                        duplicates='drop'
                    )
                    bin_labels = alt_labels
                    n_bins = alt_bins
                    break
                except ValueError:
                    continue
        
        bin_results = {}
        for bin_label in bin_labels:
            bin_data = data_copy[data_copy['_dose_bin'] == bin_label].copy()
            
            if len(bin_data) < 50:  # Skip if too few observations
                if verbose:
                    print(f"  {bin_label}: SKIPPED (too few observations: {len(bin_data)})")
                continue
            
            try:
                result_bin = conditional_did_pretest(
                    data=bin_data,
                    yname=yname,
                    tname=tname,
                    gname=gname,
                    idname=idname,
                    xformla=None,  # Unconditional within bin
                    control_group=control_group,
                    alp=alp,
                    bstrap=True,
                    cband=True,
                    biters=min(biters, 1999),  # Fewer iterations per bin
                    est_method=est_method,
                    print_details=False,
                    seed=seed,
                )
                
                bin_results[bin_label] = result_bin
                
                if verbose:
                    status = "PASS" if result_bin.passed() else "FAIL"
                    print(f"  {bin_label}: {status} "
                          f"(CvM p={result_bin.CvM_pval:.4f}, "
                          f"KS p={result_bin.KS_pval:.4f})")
            
            except Exception as e:
                if verbose:
                    print(f"  {bin_label}: ERROR ({str(e)})")
        
        if verbose:
            print()
            all_pass = all(r.passed() for r in bin_results.values())
            if all_pass:
                print("  -> All bins pass: PT holds within strata (use stratified DiD)")
            else:
                print("  -> Some bins fail: PT violated in specific dose ranges")
            print()
    
    # =========================================================================
    # TIER 4: HONESTDID SENSITIVITY BOUNDS (optional)
    # =========================================================================
    honest_result = None
    
    if run_honest_did:
        if verbose:
            print("TIER 4: HONESTDID SENSITIVITY ANALYSIS")
            print("-" * 70)
        
        # Validate inputs
        if event_study_betas is None or event_study_Sigma is None:
            raise ValueError(
                "run_honest_did=True requires event_study_betas and event_study_Sigma"
            )
        
        if numPrePeriods is None or numPostPeriods is None:
            raise ValueError(
                "run_honest_did=True requires numPrePeriods and numPostPeriods"
            )
        
        try:
            honest_result = honest_did_bounds(
                betas=event_study_betas,
                Sigma=event_study_Sigma,
                numPrePeriods=numPrePeriods,
                numPostPeriods=numPostPeriods,
                Mmax=Mmax,
                grid_points=honest_grid_points,
                seed=seed,
            )
            
            if verbose:
                print(f"Method: {honest_result.method} ({honest_result.delta_label})")
                print(f"M-grid: {honest_result.M}")
                print()
                print("Bounds by violation magnitude:")
                for M, lb, ub in zip(honest_result.M, honest_result.lb, honest_result.ub):
                    print(f"  M={M:.1f}: [{lb:.4f}, {ub:.4f}]")
                print()
        
        except Exception as e:
            if verbose:
                print(f"HonestDiD failed: {str(e)}")
            honest_result = None
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    suite = CallawayTestSuite(
        unconditional_pt=result_unc,
        conditional_pt=result_cond,
        bin_pt_tests=bin_results,
        honest_did=honest_result,
        dose_col=dose_col,
        n_bins=n_bins if bin_results else None,
    )
    
    if verbose:
        print(suite.summary())
    
    return suite


def quick_validation(
    data: pd.DataFrame,
    yname: str,
    tname: str,
    gname: str,
    idname: str,
    dose_col: str,
    **kwargs,
) -> bool:
    """
    Quick validation check: just run conditional PT test.
    
    This is a convenience function for when you just want a pass/fail
    on the identifying assumption without the full testing suite.
    
    Parameters
    ----------
    data, yname, tname, gname, idname, dose_col : see run_callaway_tests
    **kwargs : additional arguments passed to conditional_did_pretest
    
    Returns
    -------
    bool
        True if conditional PT test passes.
    
    Examples
    --------
    >>> if quick_validation(data=df, yname="y", tname="t", gname="g",
    ...                     idname="id", dose_col="dose"):
    ...     print("Proceed with estimation")
    ... else:
    ...     print("PT assumption violated")
    """
    result = conditional_did_pretest(
        data=data,
        yname=yname,
        tname=tname,
        gname=gname,
        idname=idname,
        xformla=f"~{dose_col}",
        **kwargs,
    )
    return result.passed()