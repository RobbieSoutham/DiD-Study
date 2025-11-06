"""
Pre‑trend diagnostic tests for event studies.

This module implements joint and linear tests for assessing the
parallel trends assumption in event study regressions.  The focus is
on identifying whether the lead coefficients (pre‑treatment periods)
are jointly zero (joint pretest) or follow a linear trend (slope
test).  These tests should be used for reporting rather than for
conditioning decisions【14963393938263†L7-L16】, since pre‑testing can
distort inference【14963393938263†L28-L33】.

The functions here operate on a fitted statsmodels OLS result with
cluster‑robust covariance already computed.  They return test
statistics and p‑values for the relevant hypotheses.
"""

from __future__ import annotations

from typing import Sequence, Dict, Any, Optional

import numpy as np
from scipy.stats import chi2


def joint_pretest_zero(result, pre_param_names: Sequence[str]) -> Dict[str, Any]:
    """Wald test that all specified pre‑treatment coefficients are zero.

    Parameters
    ----------
    result : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted regression result.  Must have attributes ``params`` and
        ``cov_params``.
    pre_param_names : sequence of str
        Names of coefficients corresponding to pre‑treatment (lead)
        indicators in the event study.

    Returns
    -------
    dict
        Dictionary with keys ``stat`` (chi‑square statistic), ``df``
        (degrees of freedom), ``p_value`` and ``tested`` (the subset of
        names actually found in the model).
    """
    names = [n for n in pre_param_names if n in result.params.index]
    if not names:
        raise ValueError("None of the requested pre coefficients were found.")
    b = result.params.loc[names].to_numpy(float)
    V = result.cov_params().loc[names, names].to_numpy(float)
    # symmetrise covariance
    V = 0.5 * (V + V.T)
    # invert with pseudo‑inverse for stability
    Vinv = np.linalg.pinv(V, rcond=1e-12)
    stat = float(b.T @ Vinv @ b)
    k = len(names)
    p = float(1.0 - chi2.cdf(stat, df=k))
    return {"stat": stat, "df": k, "p_value": p, "tested": names}


def linear_weights_from_event_times(pre_event_times: Sequence[int]) -> np.ndarray:
    """Construct mean‑zero unit‑norm linear weights for pre‑trend testing.

    Given a sequence of event times (negative integers), this function
    returns a vector of weights proportional to the event times, centred
    to have mean zero and scaled to unit L2 norm.  These weights are
    typically used to test for a linear pre‑trend.

    Parameters
    ----------
    pre_event_times : sequence of int
        Event times corresponding to the pre‑treatment coefficients (e.g.
        ``[-5, -4, -3, -2]``).  Do not include the omitted baseline period.

    Returns
    -------
    numpy.ndarray
        Normalised weight vector.
    """
    t = np.asarray(pre_event_times, dtype=float)
    w = t.copy()
    w -= w.mean()
    norm = np.linalg.norm(w)
    return w / (norm if norm > 0 else 1.0)


def pre_slope_test(
    result,
    pre_param_names: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    pre_event_times: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Test a linear trend in the pre‑treatment coefficients.

    This test evaluates whether a weighted sum of the pre‑treatment
    coefficients equals zero.  If ``weights`` is provided it must
    align with ``pre_param_names``.  Otherwise, if ``pre_event_times``
    are given, weights are constructed proportional to the event times
    using :func:`linear_weights_from_event_times`.

    Parameters
    ----------
    result : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted regression result with cluster‑robust covariance.
    pre_param_names : sequence of str
        Names of the pre‑treatment coefficient parameters.
    weights : sequence of float or None, optional
        Weights applied to the coefficients.  If ``None`` use
        ``pre_event_times``.
    pre_event_times : sequence of int or None, optional
        Event times corresponding to the pre‑treatment coefficients.

    Returns
    -------
    dict
        Dictionary with keys ``stat`` (chi‑square statistic), ``df`` (1),
        ``p_value``, ``contrast`` (the normalised weight vector) and
        ``tested`` (names tested).
    """
    names = [n for n in pre_param_names if n in result.params.index]
    if not names:
        raise ValueError("None of the requested pre coefficients were found.")
    b = result.params.loc[names].to_numpy(float)
    V = result.cov_params().loc[names, names].to_numpy(float)
    V = 0.5 * (V + V.T)
    # determine weights
    if weights is None:
        if pre_event_times is None or len(pre_event_times) != len(names):
            raise ValueError("Provide weights or matching pre_event_times.")
        L = linear_weights_from_event_times(pre_event_times)
    else:
        L = np.asarray(weights, dtype=float)
        if L.shape[0] != len(names):
            raise ValueError("Weights length must match number of pre coefficients.")
        L = L - L.mean()
        norm = np.linalg.norm(L)
        if norm > 0:
            L = L / norm
    # compute test statistic (Wald t^2)
    Lb = float(L @ b)
    Var_Lb = float(L @ V @ L)
    stat = 0.0 if Var_Lb <= 0 else Lb * Lb / Var_Lb
    p = float(1.0 - chi2.cdf(stat, df=1))
    return {"stat": stat, "df": 1, "p_value": p, "contrast": L, "tested": names}