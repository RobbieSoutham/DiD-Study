"""General utilities for the DiD toolkit.

This module provides helper functions that do not naturally belong to
any estimator or robustness component.  Currently it exposes a
function for selecting the wild cluster bootstrap weight type and
replication count based on the number of clusters.  It also
reexports the minimum detectable effect function from the
robustness statistics submodule for convenience.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # Optional, only needed for pretty-printing
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas always available in runtime
    pd = None  # type: ignore

# Reexport the analytic MDE function from the robustness subpackage.
from ..robustness.stats.mde import analytic_mde_from_se  # type: ignore


def choose_wcb_weights_and_B(
    G_total: int,
    G_treated: Optional[int] = None,
    B_requested: Optional[int] = None,
) -> Tuple[str, int]:
    """Choose the weight distribution and replication count for the wild cluster bootstrap.

    Simulation evidence suggests using the Webb six-point distribution when
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


def resolve_fe_terms(
    fe_terms: Optional[Sequence[str]],
    year_col: str = "Year",
    unit_col: str = "unit_id",
    *,
    fallback: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return a de-duplicated list of FE identifiers aligned with the data columns."""
    source = fe_terms if fe_terms is not None else fallback
    if source is None:
        return []

    resolved: List[str] = []
    year_key = str(year_col).lower()
    unit_key = str(unit_col).lower()

    for term in source:
        if term is None:
            continue
        raw = str(term).strip()
        if not raw:
            continue
        key = raw.lower()
        if key in {"year", year_key}:
            name = year_col
        elif key in {"unit", "unit_id", unit_key}:
            name = unit_col
        else:
            name = raw
        if name not in resolved:
            resolved.append(name)
    return resolved


__all__ = ["choose_wcb_weights_and_B", "analytic_mde_from_se", "resolve_fe_terms"]


def log_wcb_call(
    *,
    file_name: str,
    func_name: str,
    params: Optional[Dict[str, Any]] = None,
    dataframes: Optional[Dict[str, Any]] = None,
) -> None:
    """Pretty-print the arguments supplied to a WCB helper."""
    line = "=" * 72
    print(f"\n{line}")
    print(f"[WCB CALL] {file_name} -> {func_name}")
    print(line)

    if params:
        print("Parameters:")
        for key, val in params.items():
            print(f"  - {key}: {val}")

    if dataframes:
        for name, obj in dataframes.items():
            print(f"\n[{name} head]")
            if pd is not None and isinstance(obj, pd.DataFrame):
                with pd.option_context("display.max_rows", 5, "display.max_columns", 10, "display.width", 120):
                    print(obj.head().to_string(index=False))
            else:
                head_func = getattr(obj, "head", None)
                if callable(head_func):
                    try:
                        preview = head_func()
                        print(preview)
                    except Exception:
                        print(repr(obj))
                else:
                    print(repr(obj))
    print(line + "\n")


__all__.append("log_wcb_call")
