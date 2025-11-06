"""General utilities for the DiD toolkit.

This module provides helper functions that do not naturally belong to
any estimator or robustness component.  Currently it exposes a
function for selecting the wild cluster bootstrap weight type and
replication count based on the number of clusters.  It also
reexports the minimum detectable effect function from the
robustness statistics submodule for convenience.
"""

from __future__ import annotations

from typing import Optional, Tuple

# Reexport the analytic MDE function from the robustness subpackage.
from ..robustness.stats.mde import analytic_mde_from_se  # type: ignore


def choose_wcb_weights_and_B(
    G_total: int,
    G_treated: Optional[int] = None,
    B_requested: Optional[int] = None,
) -> Tuple[str, int]:
    """Choose the weight distribution and replication count for the wild cluster bootstrap.

    Simulation evidence suggests using the Webb six‑point distribution when
    the number of clusters is between five and twelve, and the
    Rademacher distribution otherwise.  The default number of
    bootstrap replications depends on the weight type: 9,999 for
    Webb and 4,999 for Rademacher, unless a specific ``B_requested`` is
    provided.

    Parameters
    ----------
    G_total : int
        Total number of clusters.
    G_treated : int or None, optional
        Number of treated clusters (included for informational purposes).
    B_requested : int or None, optional
        Desired number of bootstrap replications.  If ``None`` a
        default is chosen based on the weight type.

    Returns
    -------
    (str, int)
        A tuple ``(weight_type, B)`` where ``weight_type`` is either
        ``'webb'`` or ``'rademacher'`` and ``B`` is the number of
        bootstrap replications.
    """
    wt = "webb" if 5 <= int(G_total) <= 12 else "rademacher"
    B = int(B_requested or (9999 if wt == "webb" else 4999))
    if wt == "webb" and B < 9999:
        B = 9999
    return wt, B


__all__ = ["choose_wcb_weights_and_B", "analytic_mde_from_se"]