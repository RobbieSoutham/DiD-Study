# did_study/robustness/honest_did.py
# COMPLETE VERSION - November 14, 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from did_study.robustness.r_interface import set_r_seeds


@dataclass
class HonestDiDResult:
    """Container for HonestDiD relative-magnitude bounds on a scalar θ."""
    M: np.ndarray  # grid of Mbar values
    lb: np.ndarray  # lower bounds
    ub: np.ndarray  # upper bounds
    method: str  # e.g. "C-LF"
    delta_label: str  # e.g. "DeltaRM"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "M": self.M,
            "lb": self.lb,
            "ub": self.ub,
            "method": self.method,
            "Delta": self.delta_label,
        }


def honest_did_bounds(
    betas: np.ndarray,
    Sigma: np.ndarray,
    numPrePeriods: int,
    numPostPeriods: int,
    Mmax: Optional[float] = None,
    grid_points: int = 10,
    seed: Optional[int] = 123,
    l_vec: Optional[np.ndarray] = None,
) -> HonestDiDResult:
    """
    Compute HonestDiD Δ^RM ("relative magnitudes") sensitivity bounds for θ.

    Parameters
    ----------
    betas : np.ndarray
        Event-study coefficients, ordered [pre_periods, post_periods].
        Length = numPrePeriods + numPostPeriods.
        Example: [β_{-5}, β_{-4}, β_{-3}, β_{-2}, β_0, β_1, ..., β_8]

    Sigma : np.ndarray
        Full covariance matrix of event-study coefficients (cluster-robust).
        Shape = (numPrePeriods + numPostPeriods, numPrePeriods + numPostPeriods).

    numPrePeriods : int
        Number of pre-treatment periods. Example: if leads are [-5, -4, -3, -2], then numPrePeriods = 4.

    numPostPeriods : int
        Number of post-treatment periods. Example: if lags are [0, 1, ..., 8], then numPostPeriods = 9.

    Mmax : float, optional
        Maximum relative magnitude threshold. Default 2.0.
        M=0: no parallel-trend violation
        M=1: post-violation ≤ pre-violation
        M=2: post-violation can be up to 2x pre-violation (more conservative)

    grid_points : int
        Number of grid points between 0 and Mmax.

    seed : int, optional
        RNG seed for reproducibility.

    l_vec : np.ndarray, optional
        Length numPostPeriods. Weights for post-treatment effects: θ = l_vec' * τ_post.
        If None, HonestDiD defaults to picking the first post-period.

    Returns
    -------
    HonestDiDResult
        M, lb, ub arrays and metadata.

    Raises
    ------
    ValueError
        If dimensions don't match.
    RuntimeError
        If HonestDiD R call fails.
    """

    # Validate inputs
    if not isinstance(betas, np.ndarray):
        betas = np.asarray(betas, dtype=float)
    if not isinstance(Sigma, np.ndarray):
        Sigma = np.asarray(Sigma, dtype=float)

    numTotal = numPrePeriods + numPostPeriods

    if betas.size != numTotal:
        raise ValueError(
            f"betas length {betas.size} != numPrePeriods + numPostPeriods "
            f"({numPrePeriods} + {numPostPeriods} = {numTotal})"
        )

    expected_shape = (numTotal, numTotal)
    if Sigma.shape != expected_shape:
        raise ValueError(
            f"Sigma shape {Sigma.shape} != expected {expected_shape}"
        )

    # Choose Mmax
    if Mmax is None:
        Mmax = 2.0

    if Mmax < 0:
        raise ValueError(f"Mmax must be non-negative; got {Mmax}.")

    M_grid = np.linspace(0.0, Mmax, num=int(grid_points))

    # Validate l_vec if provided
    if l_vec is not None:
        l_vec = np.asarray(l_vec, dtype=float)
        # CRITICAL FIX: Check against numPostPeriods (int), not a list!
        if l_vec.size != numPostPeriods:
            raise ValueError(
                f"l_vec must have length numPostPeriods={numPostPeriods}, "
                f"got {l_vec.size}."
            )
        if not np.isclose(l_vec.sum(), 1.0, atol=1e-6):
            print(f"[Warning] l_vec sums to {l_vec.sum():.6f}, not 1.0")

    # Call HonestDiD in R
    HonestDiD = importr("HonestDiD")
    set_r_seeds(seed)

    R_beta = ro.FloatVector(betas.tolist())
    R_Sigma = ro.r["matrix"](
        ro.FloatVector(Sigma.ravel(order="C")),
        nrow=Sigma.shape[0],
    )
    R_M = ro.FloatVector(M_grid.tolist())
    R_l_vec = None

    if l_vec is not None:
        R_l_vec = ro.FloatVector(l_vec.tolist())

    print("=" * 72)
    print("[HonestDiD CALL] -> createSensitivityResults_relativeMagnitudes")
    print("=" * 72)
    print(f"Parameters:")
    print(f"  - betahat length: {len(betas)}")
    print(f"  - Sigma shape: {Sigma.shape}")
    print(f"  - numPrePeriods: {numPrePeriods}")
    print(f"  - numPostPeriods: {numPostPeriods}")
    print(f"  - Mbar grid: {M_grid}")
    if R_l_vec is not None:
        print(f"  - l_vec: {l_vec}")
    print()

    kwargs: Dict[str, Any] = dict(
        betahat=R_beta,
        sigma=R_Sigma,
        numPrePeriods=numPrePeriods,
        numPostPeriods=numPostPeriods,
        bound="deviation from parallel trends",
        method="C-LF",
        Mbarvec=R_M,
        seed=int(seed or 0),
    )

    raise Exception(sigma)

    if R_l_vec is not None:
        kwargs["l_vec"] = R_l_vec

    try:
        R_bounds = HonestDiD.createSensitivityResults_relativeMagnitudes(**kwargs)
    except Exception as e:
        raise RuntimeError(f"HonestDiD R call failed: {e}") from e

    # Parse results
    try:
        df_bounds = pandas2ri.rpy2py(R_bounds)

        if not isinstance(df_bounds, pd.DataFrame):
            raise TypeError(f"Expected DataFrame; got {type(df_bounds)}")

        colmap = {c.lower(): c for c in df_bounds.columns}

        lb_col = colmap.get("lb")
        ub_col = colmap.get("ub")
        m_col = colmap.get("mbar") or colmap.get("m")
        method_col = colmap.get("method")
        delta_col = colmap.get("delta") or colmap.get("deltarm")

        if lb_col is None or ub_col is None or m_col is None:
            raise KeyError(
                f"HonestDiD results missing required columns. "
                f"Have: {list(df_bounds.columns)}"
            )

        lb = df_bounds[lb_col].to_numpy(dtype=float)
        ub = df_bounds[ub_col].to_numpy(dtype=float)
        M = df_bounds[m_col].to_numpy(dtype=float)

        method = (
            str(df_bounds[method_col].iloc[0])
            if method_col is not None and len(df_bounds) > 0
            else "C-LF"
        )

        delta_label = (
            str(df_bounds[delta_col].iloc[0])
            if delta_col is not None and len(df_bounds) > 0
            else "DeltaRM"
        )

        print(f"HonestDiD Results (Δ^RM):")
        print(f"  M-grid: {M}")
        print(f"  Lower bounds: {lb}")
        print(f"  Upper bounds: {ub}")
        print()

        return HonestDiDResult(M=M, lb=lb, ub=ub, method=method, delta_label=delta_label)

    except Exception as e:
        raise RuntimeError(f"Failed to parse HonestDiD output: {e}") from e
