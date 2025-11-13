# did_study/robustness/honest_did.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from did_study.robustness.r_interface import set_r_seeds


@dataclass
class HonestDiDResult:
    """Container for HonestDiD relative-magnitude bounds on a scalar θ."""

    M: np.ndarray          # grid of Mbar values
    lb: np.ndarray         # lower bounds
    ub: np.ndarray         # upper bounds
    method: str            # e.g. "C-LF"
    delta_label: str       # e.g. "DeltaRM"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "M": self.M,
            "lb": self.lb,
            "ub": self.ub,
            "method": self.method,
            "Delta": self.delta_label,
        }


def sanitize_es_for_honestdid(
    es_df: pd.DataFrame,
    pre_periods: Sequence[int],
    post_periods: Sequence[int],
    beta_col: str = "att",
    se_col: str = "se",
) -> SimpleNamespace:
    """
    Extract the event-study coefficient vector and covariance matrix in the
    format expected by HonestDiD.

    Following Rambachan & Roth (2023), we assume betahat is stacked as:

        betahat = (β_pre, τ_post),

    where β_pre are pre-treatment event-time coefficients and τ_post are
    post-treatment event-time coefficients, both ordered chronologically.

    Parameters
    ----------
    es_df : DataFrame
        Event-study results with at least columns: 'event_time', beta_col, se_col.
    pre_periods : sequence of int
        Pre-treatment event times (e.g. [-5, -4, -3, -2]).
    post_periods : sequence of int
        Post-treatment event times (e.g. [0, 1, 2, 3, 4, 5]).
    beta_col, se_col : str
        Column names for the point estimates and standard errors.

    Returns
    -------
    SimpleNamespace
        Fields:
          - betas: np.ndarray of length numPre + numPost
          - Sigma: 2D np.ndarray covariance matrix
          - numPrePeriods, numPostPeriods: ints
          - pre_idx, post_idx: index arrays into the stacked beta vector
    """
    df = es_df.copy()

    if "event_time" not in df.columns:
        raise ValueError("es_df must contain an 'event_time' column for HonestDiD.")

    # Ensure sorted by event_time for stable ordering: negatives first, then non-negatives
    df = df.sort_values("event_time").reset_index(drop=True)

    # Keep only the event times we designate as pre/post (drop reference periods)
    mask = df["event_time"].isin(list(pre_periods) + list(post_periods))
    df = df.loc[mask].copy()

    if df.empty:
        raise ValueError("No event-study rows remain after filtering to pre/post periods.")

    pre_idx = np.where(df["event_time"].isin(pre_periods))[0]
    post_idx = np.where(df["event_time"].isin(post_periods))[0]

    num_pre = len(pre_idx)
    num_post = len(post_idx)

    if num_pre == 0 or num_post == 0:
        raise ValueError(
            f"Need at least one pre and one post period for HonestDiD; "
            f"got numPre={num_pre}, numPost={num_post}."
        )

    # Betas vector: stacked in chronological order (pre, then post)
    betas = df[beta_col].to_numpy(dtype=float)

    # Covariance: here we approximate with diag(se^2). If you have a full
    # covariance matrix from the event-study regression, you can plug it in
    # here instead to match the richest implementation in the literature.
    if se_col not in df.columns:
        raise ValueError(
            f"es_df must contain column '{se_col}' for HonestDiD (standard errors)."
        )
    ses = df[se_col].to_numpy(dtype=float)
    Sigma = np.diag(ses ** 2)

    return SimpleNamespace(
        betas=betas,
        Sigma=Sigma,
        numPrePeriods=num_pre,
        numPostPeriods=num_post,
        pre_idx=pre_idx,
        post_idx=post_idx,
    )


def honest_did_bounds(
    es_df: pd.DataFrame,
    pre_periods: Sequence[int],
    post_periods: Sequence[int],
    beta_col: str = "att",
    se_col: str = "se",
    Mmax: Optional[float] = None,
    grid_points: int = 10,
    seed: Optional[int] = 123,
    l_vec: Optional[np.ndarray] = None,
) -> HonestDiDResult:
    """
    Compute HonestDiD Δ^RM (“relative magnitudes”) sensitivity bounds for θ.

    We call HonestDiD::createSensitivityResults_relativeMagnitudes with:

        - bound = "deviation from parallel trends"  (Δ^RM)
        - method = "C-LF"                           (Conley et al. local projections)
        - Mbarvec = grid of Mbar in [0, Mmax]
        - l_vec = user-specified weights for the scalar parameter
                 θ = l_vec' * τ_post (length = numPostPeriods)

    Parameters
    ----------
    es_df : DataFrame
        Event-study table (one row per event_time).
    pre_periods, post_periods : sequences of int
        Pre- and post-treatment event times used in the PTA.
    beta_col, se_col : str
        Column names of estimates and standard errors in es_df.
    Mmax : float, optional
        Maximum relative magnitude (Mbar) to consider. If None, defaults to 2.0.
        This is a *substantive* choice: higher Mmax allows larger deviations
        from parallel trends (more conservative bounds).
    grid_points : int
        Number of grid points between 0 and Mmax (inclusive).
    seed : int, optional
        Seed passed through to the R RNG to ensure reproducibility.
    l_vec : np.ndarray, optional
        Length numPostPeriods. If provided, defines θ = l_vec' * τ_post.
        If None, HonestDiD defaults to a basis vector picking out the first
        post-treatment period.

    Returns
    -------
    HonestDiDResult
        Contains arrays of M, lower and upper bounds, and metadata.
    """
    # 1) Extract betas and covariance in the format HonestDiD expects
    es = sanitize_es_for_honestdid(
        es_df=es_df,
        pre_periods=pre_periods,
        post_periods=post_periods,
        beta_col=beta_col,
        se_col=se_col,
    )

    betas = es.betas
    Sigma = es.Sigma

    # 2) Choose Mmax if not provided (substantive choice, not data-driven)
    if Mmax is None:
        Mmax = 2.0
    if Mmax <= 0:
        raise ValueError(f"Mmax must be positive; got {Mmax}.")

    # Grid over [0, Mmax]
    M_grid = np.linspace(0.0, Mmax, num=grid_points)

    # 3) Call HonestDiD in R
    HonestDiD = importr("HonestDiD")

    # Synchronise RNG state for reproducibility
    set_r_seeds(seed)

    # Convert to R objects
    R_beta = ro.FloatVector(betas.tolist())
    R_Sigma = ro.r["matrix"](
        ro.FloatVector(Sigma.ravel(order="C")),
        nrow=Sigma.shape[0],
    )
    R_M = ro.FloatVector(M_grid.tolist())

    R_l_vec = None
    if l_vec is not None:
        l_vec = np.asarray(l_vec, dtype=float)
        if l_vec.size != es.numPostPeriods:
            raise ValueError(
                f"l_vec must have length numPostPeriods={es.numPostPeriods}, "
                f"got {l_vec.size}."
            )
        R_l_vec = ro.FloatVector(l_vec.tolist())

    # Optional debug logging
    print("=" * 72)
    print("[HonestDiD CALL] -> createSensitivityResults_relativeMagnitudes")
    print("=" * 72)
    print("Parameters:")
    print(f"  - betahat: {R_beta}")
    print(f"  - R_Sigma: {R_Sigma}")
    print(f"  - n_pre: {es.numPrePeriods}")
    print(f"  - n_post: {es.numPostPeriods}")
    print(f"  - Mbar grid: {R_M}")

    kwargs: Dict[str, Any] = dict(
        betahat=R_beta,
        sigma=R_Sigma,
        numPrePeriods=es.numPrePeriods,
        numPostPeriods=es.numPostPeriods,
        bound="deviation from parallel trends",
        method="C-LF",
        Mbarvec=R_M,
        seed=int(seed or 0),
    )
    if R_l_vec is not None:
        kwargs["l_vec"] = R_l_vec

    R_bounds = HonestDiD.createSensitivityResults_relativeMagnitudes(**kwargs)

    # 4) Parse the tibble/data.frame returned by HonestDiD
    try:
        df_bounds = pandas2ri.rpy2py(R_bounds)

        if not isinstance(df_bounds, pd.DataFrame):
            raise TypeError(
                f"Expected HonestDiD to return a DataFrame; got {type(df_bounds)}."
            )

        colmap = {c.lower(): c for c in df_bounds.columns}
        lb_col = colmap.get("lb")
        ub_col = colmap.get("ub")
        m_col = colmap.get("mbar") or colmap.get("m")
        method_col = colmap.get("method")
        delta_col = colmap.get("delta") or colmap.get("deltarm")

        if lb_col is None or ub_col is None or m_col is None:
            raise KeyError(
                f"HonestDiD results missing required columns "
                f"(have: {list(df_bounds.columns)})"
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

        return HonestDiDResult(M=M, lb=lb, ub=ub, method=method, delta_label=delta_label)

    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse HonestDiD relative-magnitude output: {e}") from e
