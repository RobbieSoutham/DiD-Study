from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict

import numpy as np
import pandas as pd

from ..helpers.config import StudyConfig
from ..helpers.preparation import PanelData


@dataclass
class DifferencesAttResult:
    """
    Result container for the Callaway–Sant'Anna-style ATT using the
    Python `differences` package (ATTgt object), plus an overall ATT^o.
    """

    att_overall: float | np.nan
    se_overall: float | np.nan
    p_overall: float | np.nan
    used: pd.DataFrame
    attgt_obj: Any
    att_gt_df: pd.DataFrame


def _safe_isfinite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _build_differences_input(panel: PanelData, config: StudyConfig) -> pd.DataFrame:
    """
    Prepare input DataFrame expected by `differences` package ATTgt API.

    Columns produced:
      - unit_id: cluster id
      - t: time (from config.year_col)
      - g: first treatment year (NaN for never-treated)
      - D: absorbing treatment at time t (1 if treated and t >= g; else 0)
      - y: outcome (PanelData.outcome_name)
    """
    df = panel.panel.copy()
    year_col = getattr(config, "year_col", "Year")
    outcome = getattr(panel, "outcome_name", None)

    need = ["unit_id", year_col, "treated_ever", "g"]
    if outcome:
        need.append(outcome)
    used = df.dropna(subset=[c for c in need if c in df.columns]).copy()

    if used.empty or ("unit_id" in used and used["unit_id"].nunique() < 2):
        return used.assign(t=np.nan, D=np.nan, y=np.nan)

    used = used.sort_values(["unit_id", year_col]).copy()
    used["t"] = used[year_col]

    # absorbing treatment indicator
    if {"treated_ever", "g"} <= set(used.columns):
        used["D"] = np.where(
            (used["treated_ever"].astype(int) == 1) & (used["t"] >= used["g"]),
            1,
            0,
        )
    else:
        # fallback: use provided treated_now if present
        used["D"] = used.get("treated_now", 0).astype(int)

    # outcome as 'y'
    if outcome and outcome in used.columns:
        used["y"] = used[outcome].astype(float)
    else:
        used["y"] = np.nan

    return used


def _import_attgt_class() -> Optional[Any]:
    """Attempt to import an ATTgt-like class from the differences package.

    We try a few plausible module paths to be robust to minor API variations.
    Returns the class if import succeeds, else None.
    """
    try:
        from differences import ATTgt  # type: ignore
        return ATTgt
    except Exception:
        pass
    # Alternative import paths seen in earlier/other versions
    for modpath, name in [
        ("differences.attgt", "ATTgt"),
        ("differences.att", "ATTgt"),
        ("differences", "ATTG"),
    ]:
        try:
            import importlib

            m = importlib.import_module(modpath)
            if hasattr(m, name):
                return getattr(m, name)
        except Exception:
            continue
    return None


def _fit_attgt(attgt_obj: Any, *, use_formula: bool, covariates: Optional[list[str]] = None) -> Any:
    """Call the appropriate fit method with either a formula or default fit.

    If the object exposes a .fit(formula=...) API we use it; otherwise try
    common fallbacks such as .fit() without args.
    """
    try:
        import inspect

        fit = getattr(attgt_obj, "fit", None)
        if fit is None:
            # Sometimes computation is done at construction time
            return attgt_obj

        sig = inspect.signature(fit)
        if use_formula and any(p.name == "formula" for p in sig.parameters.values()):
            formula_rhs = "1"
            if covariates:
                # include controls if provided
                formula_rhs = "1 + " + " + ".join([str(c) for c in covariates])
            return fit(formula=f"y ~ {formula_rhs}")
        else:
            # minimal call
            return fit()
    except Exception:
        # last resort: try a bare call
        try:
            return attgt_obj.fit()
        except Exception:
            return attgt_obj


def _aggregate_overall(attgt_obj: Any) -> Dict[str, float]:
    """Try to extract an overall ATT and its inference from the ATTgt object.

    Returns a dict with keys {"att", "se", "p"} when available; missing
    entries are set to np.nan.
    """
    out = {"att": np.nan, "se": np.nan, "p": np.nan}
    # Common pattern: obj.aggregate("overall") or obj.aggregate("simple")
    for key in ("overall", "simple"):
        try:
            agg = attgt_obj.aggregate(key)  # type: ignore[attr-defined]
            # Heuristics: allow scalar, tuple, or dict-like
            if isinstance(agg, (list, tuple)) and len(agg) >= 1:
                out["att"] = float(agg[0])
                if len(agg) >= 2:
                    try:
                        out["se"] = float(agg[1])
                    except Exception:
                        pass
                if len(agg) >= 3:
                    try:
                        out["p"] = float(agg[2])
                    except Exception:
                        pass
                return out
            if isinstance(agg, dict):
                # prefer common keys
                for k_src, k_dst in [("att", "att"), ("estimate", "att"), ("coef", "att"), ("se", "se"), ("p", "p"), ("p_value", "p")]:
                    if k_src in agg:
                        try:
                            out[k_dst] = float(agg[k_src])
                        except Exception:
                            pass
                return out
            # DataFrame/result object: try attributes
            for attr in ["att", "estimate", "coef"]:
                if hasattr(agg, attr):
                    try:
                        out["att"] = float(getattr(agg, attr))
                    except Exception:
                        pass
            for attr in ["se", "stderr", "std_err", "std"]:
                if hasattr(agg, attr):
                    try:
                        out["se"] = float(getattr(agg, attr))
                    except Exception:
                        pass
            for attr in ["p", "p_value", "pval"]:
                if hasattr(agg, attr):
                    try:
                        out["p"] = float(getattr(agg, attr))
                    except Exception:
                        pass
            return out
        except Exception:
            continue

    # Some versions expose a .summary() with overall
    try:
        summ = attgt_obj.summary()  # type: ignore[attr-defined]
        if isinstance(summ, dict):
            for k_src, k_dst in [("overall", "att"), ("se", "se"), ("p", "p"), ("p_value", "p")]:
                if k_src in summ:
                    try:
                        out[k_dst] = float(summ[k_src])
                    except Exception:
                        pass
        return out
    except Exception:
        pass

    return out


def _extract_attgt_table(attgt_obj: Any) -> pd.DataFrame:
    """Try to obtain the cohort×time ATTgt table as a pandas DataFrame.

    We check a few common attribute names and coerce columns to a standard
    schema: ['g','t','att','se'] when available.
    """
    cand_attrs = ["att_gt", "attgt", "results", "table", "att_table"]
    df = None
    for name in cand_attrs:
        try:
            obj = getattr(attgt_obj, name)
        except Exception:
            continue
        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
            break
    if df is None:
        return pd.DataFrame(columns=["g", "t", "att", "se"])  # empty

    # Normalize column names
    cols = {c.lower(): c for c in df.columns if isinstance(c, str)}
    def pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    col_g = pick("g", "cohort", "first_treat", "first_treatment", "group")
    col_t = pick("t", "time", "period", "year")
    col_att = pick("att", "estimate", "coef", "effect")
    col_se = pick("se", "stderr", "std_err", "std")

    out = pd.DataFrame()
    if col_g is not None:
        out["g"] = df[col_g].astype(float)
    if col_t is not None:
        out["t"] = df[col_t].astype(float)
    if col_att is not None:
        out["att"] = df[col_att].astype(float)
    if col_se is not None:
        out["se"] = df[col_se].astype(float)

    return out


def estimate_att_o_differences(
    panel: PanelData,
    config: StudyConfig,
) -> DifferencesAttResult:
    """
    Estimate ATT^o using the Python `differences` package (Callaway–Sant'Anna style).

    Returns a DifferencesAttResult. If the package is not installed or the
    data is not suitable, returns NaNs and an empty att_gt_df.
    """
    # Prepare input
    used = _build_differences_input(panel, config)
    if used.empty or used["unit_id"].nunique() < 2 or used["y"].isna().all():
        return DifferencesAttResult(np.nan, np.nan, np.nan, used, None, pd.DataFrame())

    # Try to import ATTgt
    ATTgtClass = None
    try:
        ATTgtClass = _import_attgt_class()
    except Exception:
        ATTgtClass = None

    if ATTgtClass is None:
        print("[differences] Package not available. Skipping differences-based ATT^o.")
        return DifferencesAttResult(np.nan, np.nan, np.nan, used, None, pd.DataFrame())

    # Instantiate ATTgt with robust handling of parameter names
    # Common kwargs: cohort_name, time_name, unit_name, treatment_name
    kwargs: Dict[str, Any] = {}
    for k in ["cohort_name", "cohort_col", "cohort"]:
        kwargs.setdefault(k, None)
    for k in ["time_name", "time_col", "time"]:
        kwargs.setdefault(k, None)
    for k in ["unit_name", "unit_col", "unit"]:
        kwargs.setdefault(k, None)
    for k in ["treatment_name", "treat_col", "treatment", "D_name"]:
        kwargs.setdefault(k, None)

    # Minimal set we know we have
    # We'll introspect the constructor to only pass accepted kwargs
    try:
        import inspect

        sig = inspect.signature(ATTgtClass)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    # Fill accepted names only
    if "cohort_name" in params:
        kwargs["cohort_name"] = "g"
    elif "cohort_col" in params:
        kwargs["cohort_col"] = "g"
    elif "cohort" in params:
        kwargs["cohort"] = "g"

    if "time_name" in params:
        kwargs["time_name"] = "t"
    elif "time_col" in params:
        kwargs["time_col"] = "t"
    elif "time" in params:
        kwargs["time"] = "t"

    if "unit_name" in params:
        kwargs["unit_name"] = "unit_id"
    elif "unit_col" in params:
        kwargs["unit_col"] = "unit_id"
    elif "unit" in params:
        kwargs["unit"] = "unit_id"

    if "treatment_name" in params:
        kwargs["treatment_name"] = "D"
    elif "treat_col" in params:
        kwargs["treat_col"] = "D"
    elif "treatment" in params:
        kwargs["treatment"] = "D"
    elif "D_name" in params:
        kwargs["D_name"] = "D"

    # Data argument name varies: try to detect
    data_kwargs: Dict[str, Any] = {}
    if "data" in params:
        data_kwargs["data"] = used
    elif "df" in params:
        data_kwargs["df"] = used
    else:
        # If constructor doesn't take data, we will pass later to fit if possible
        pass

    try:
        attgt = ATTgtClass(**data_kwargs, **{k: v for k, v in kwargs.items() if k in params and v is not None})
    except TypeError:
        # try with only the essentials
        essentials = {k: v for k, v in {"cohort_name": "g", "time_name": "t", "unit_name": "unit_id", "treatment_name": "D"}.items() if k in params}
        try:
            attgt = ATTgtClass(**data_kwargs, **essentials)
        except Exception as e:
            print("[differences] Failed to construct ATTgt object:", repr(e))
            return DifferencesAttResult(np.nan, np.nan, np.nan, used, None, pd.DataFrame())
    except Exception as e:
        print("[differences] Failed to construct ATTgt object:", repr(e))
        return DifferencesAttResult(np.nan, np.nan, np.nan, used, None, pd.DataFrame())

    # Fit: prefer formula if supported; include covariates if any
    covariates = list(getattr(panel, "covar_cols_used", []) or [])
    _fit_attgt(attgt, use_formula=True, covariates=covariates)

    # Aggregate overall
    agg = _aggregate_overall(attgt)
    att_overall = float(agg.get("att", np.nan))
    se_overall = float(agg.get("se", np.nan))
    p_overall = float(agg.get("p", np.nan))

    # ATTgt table
    att_gt_df = _extract_attgt_table(attgt)
    # ensure standard columns exist
    for req in ["g", "t", "att"]:
        if req not in att_gt_df.columns:
            att_gt_df[req] = np.nan

    return DifferencesAttResult(att_overall, se_overall, p_overall, used, attgt, att_gt_df)

