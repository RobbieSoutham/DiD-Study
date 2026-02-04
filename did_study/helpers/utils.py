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

import numpy as np

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


def tidy_differences_event_agg_df(df_event: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a `differences.ATTgt.aggregate('event')` result (or its CSV export)
    into a tidy DataFrame with columns: ['event_time', 'beta', 'se', 'lo', 'hi'].

    Handles:
      - The in-memory result from `att_gt.aggregate('event', boot_iterations=...)`
        with MultiIndex columns and `relative_period` as an index level.
      - CSV exports like test.csv, where pandas writes stacked header rows.
    """
    df = df_event.copy()

    # ---------- Case 1: CSV export like test.csv (stacked header) ----------
    # Detect by an "Unnamed: 0" column + a 'relative_period' marker on row 2.
    if (
        isinstance(df.columns[0], str)
        and df.columns[0].startswith("Unnamed")
        and df.shape[0] >= 3
        and str(df.iloc[2, 0]).lower() == "relative_period"
    ):
        # Data rows start at index 3; first column holds relative_period values.
        data = df.iloc[3:].copy()

        # Row 1 encodes ATT / std_error / lower / upper / zero_not_in_cband.
        header1 = df.iloc[1].tolist()

        # Build new column names: first is 'relative_period', then from header1.
        col_names = ["relative_period"]
        for i in range(1, len(data.columns)):
            h = header1[i]
            if isinstance(h, str) and h == h:
                col_names.append(h)
            else:
                col_names.append(f"col_{i}")
        data.columns = col_names

        # Recurse: next pass treats this as a "normal" event-agg frame.
        return tidy_differences_event_agg_df(data)

    # ---------- Case 2: in-memory event_agg DataFrame ----------
    # Flatten columns (works for simple and MultiIndex columns).
    col_tuples = []
    for col in df.columns:
        if isinstance(col, tuple):
            col_tuples.append(tuple(str(c) for c in col))
        else:
            col_tuples.append((str(col),))

    # Map last-level name (lower-case) -> column index (first occurrence wins).
    last_map: dict[str, int] = {}
    for idx, tup in enumerate(col_tuples):
        last = tup[-1].lower()
        if last not in last_map:
            last_map[last] = idx

    def _get_col(last_name_candidates):
        """Return the first column whose final level matches any of the candidates."""
        for key in last_name_candidates:
            idx = last_map.get(key)
            if idx is not None:
                return df.iloc[:, idx]
        return None

    # --- Event time (relative period) ---
    # Try as a column first.
    event_series = _get_col(["relative_period", "event_time", "tau"])

    # If not found as column, fall back to index.
    if event_series is None:
        idx = df.index
        if isinstance(idx, pd.MultiIndex):
            names = [(name.lower() if isinstance(name, str) else "") for name in idx.names]
            if "relative_period" in names:
                level = names.index("relative_period")
                event_series = pd.Index(idx.get_level_values(level))
        else:
            if isinstance(idx.name, str) and idx.name.lower() in ("relative_period", "event_time", "tau"):
                event_series = pd.Series(idx)

    if event_series is None:
        raise ValueError(
            "Could not find event-time information (relative_period / event_time / tau) "
            "in either columns or index."
        )

    # --- ATT / effect ---
    att_series = _get_col(["att", "estimate", "effect", "coef"])
    if att_series is None:
        raise ValueError(
            "Could not find ATT/effect column in event aggregation "
            "(ATT / estimate / effect / coef)."
        )

    # --- SE and CI bounds ---
    se_series = _get_col(["std_error", "std.error", "se"])
    lo_series = _get_col(["lower", "lb", "ci_lower"])
    hi_series = _get_col(["upper", "ub", "ci_upper"])

    # Use numpy arrays to avoid index-alignment surprises.
    evt_vals = pd.to_numeric(np.asarray(event_series), errors="coerce")
    att_vals = pd.to_numeric(np.asarray(att_series), errors="coerce")

    out = pd.DataFrame()
    out["event_time"] = evt_vals
    out["beta"] = att_vals

    if se_series is not None:
        se_vals = pd.to_numeric(np.asarray(se_series), errors="coerce")
        out["se"] = se_vals
    else:
        out["se"] = np.nan

    if lo_series is not None and hi_series is not None:
        lo_vals = pd.to_numeric(np.asarray(lo_series), errors="coerce")
        hi_vals = pd.to_numeric(np.asarray(hi_series), errors="coerce")
        out["lo"] = lo_vals
        out["hi"] = hi_vals
    elif se_series is not None:
        out["lo"] = out["beta"] - 1.96 * out["se"]
        out["hi"] = out["beta"] + 1.96 * out["se"]
    else:
        out["lo"] = np.nan
        out["hi"] = np.nan

    out = out.dropna(subset=["event_time", "beta"])
    out = out.sort_values("event_time").reset_index(drop=True)
    return out