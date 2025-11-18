# did_study/robustness/r_interface.py
from __future__ import annotations
from typing import Any, Sequence, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_rpy2():
    """Import rpy2 lazily so the package can be imported without R installed."""

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.vectors import FloatVector
    from rpy2.robjects.packages import importr

    return ro, pandas2ri, localconverter, FloatVector, importr


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def set_r_seeds(seed: Optional[int]) -> None:
    """Set both base R and dqrng seeds when available."""
    if seed is None:
        return
    s = int(seed)
    ro, _, _, _, _ = _load_rpy2()
    try:
        ro.r(f"set.seed({s})")
    except Exception:
        pass
    try:
        ro.r("if (requireNamespace('dqrng', quietly=TRUE)) "
             "dqrng::dqset.seed(%d)" % s)
    except Exception:
        pass


def _sanitize_varlist(df: pd.DataFrame, items: Sequence[str] | None) -> List[str]:
    """Return only those names that are *column names* of df."""
    cols = set(df.columns)
    out: List[str] = []
    for it in (items or []):
        if isinstance(it, pd.Series):
            nm = str(it.name) if it.name is not None else ""
        else:
            nm = str(getattr(it, "name", it) or "").strip()
        if not nm:
            continue
        # IMPORTANT: do NOT split on '+' – we only want bare column names
        if nm in cols and nm not in out:
            out.append(nm)
    return out


def _sanitize_fe_list(df: pd.DataFrame, items: Sequence[str] | None) -> List[str]:
    """Fixed-effect arguments must be valid column names present in df."""
    cols = set(df.columns)
    out: List[str] = []
    for it in (items or []):
        nm = str(it).strip()
        if nm and nm in cols and nm not in out:
            out.append(nm)
    return out


def build_fixest_formula(outcome: str,
                         regressors: Sequence[str] | None,
                         fe: Sequence[str] | None) -> str:
    """Construct a fixest formula of the form: outcome ~ x1 + x2 | fe1 + fe2"""
    rhs = " + ".join(map(str, regressors)) if regressors else "1"
    fe_rhs = " + ".join(map(str, fe)) if fe else ""
    fe_part = f" | {fe_rhs}" if fe_rhs else ""
    return f"{outcome} ~ {rhs}{fe_part}"


# ---------------------------------------------------------------------
# fixest fit factory (shared by WCB callers)
# ---------------------------------------------------------------------

def make_feols_in_r(df: pd.DataFrame,
                    outcome: str,
                    regressors: Sequence[str] | None,
                    fe: Sequence[str] | None):
    """Fit fixest::feols in R on a sanitized sub-dataframe.

    We keep the sub-dataframe in Python so that clustid lookups are
    well-defined (but boottest() itself only needs the fitted object).
    """
    ro, pandas2ri, localconverter, _, importr = _load_rpy2()
    fixest = importr("fixest")

    used_reg = _sanitize_varlist(df, regressors)
    used_fe = _sanitize_fe_list(df, fe)

    keep_cols = [c for c in {outcome, *used_reg, *used_fe} if c in df.columns]
    sub = df.loc[:, keep_cols].copy()

    # Make sure regressors are numeric where that is appropriate
    for c in used_reg:
        if c in sub.columns and not np.issubdtype(sub[c].dtype, np.number):
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # Fixed-effects and clustering variables should be treated as
    # *coded* factors in R. fwildclusterboot sometimes constructs
    # expressions like `cluster1 + cluster2` internally; if these
    # are characters this yields `non-numeric argument to binary
    # operator`. We therefore coerce non-numeric FE columns to
    # integer codes (via pandas categorical), which map one-to-one
    # to levels but are safe to add.
    for c in used_fe:
        if c in sub.columns and not np.issubdtype(sub[c].dtype, np.number):
            sub[c] = sub[c].astype("category").cat.codes

    fml_str = build_fixest_formula(outcome, used_reg, used_fe)

    with localconverter(pandas2ri.converter):
        r_df = ro.conversion.py2rpy(sub)

    fml = ro.r(f"as.formula('{fml_str}')")
    fit = fixest.feols(fml, data=r_df)
    return fit, sub, fml_str



# ---------------------------------------------------------------------
# fwildclusterboot wrappers (boottest / mboottest for fixest)
# ---------------------------------------------------------------------

def _probe_boottest_formals(fn_name: str) -> List[str]:
    """Return the list of formal argument names for fwildclusterboot::<fn_name>.

    We only care about whether 'clustid' and 'type' exist (they do in
    fwildclusterboot >= 0.13).
    """
    ro, _, _, _, _ = _load_rpy2()
    try:
        return list(ro.r(f"names(formals(fwildclusterboot::{fn_name}))"))
    except Exception:
        return []


def _sanitize_clustid_names(df: pd.DataFrame,
                            clustid: Sequence[str] | None) -> List[str]:
    """Return a list of *column names* to be used as clustid.

    We do **not** pass actual cluster values here – only the *names*,
    which fwildclusterboot will look up in the model's data.
    """
    if not clustid:
        return []
    cols = set(df.columns)
    out: List[str] = []
    for c in clustid:
        if isinstance(c, pd.Series):
            nm = c.name
        else:
            nm = str(c).strip()
        if nm and nm in cols and nm not in out:
            out.append(nm)
    return out


def boottest_fixest(*,
                    fit,
                    df: pd.DataFrame,
                    param: str,
                    B: int,
                    clustid: Sequence[str] | None,
                    type_: str,
                    impose_null: bool,
                    seed: Optional[int]):
    """Call fwildclusterboot::boottest() for a fixest object.

    Tailored for fwildclusterboot 0.14.3:
      boottest(fixest_fit, param = 'x', B = 9999,
               clustid = 'cluster_col', type = 'rademacher', ...)
    """
    ro, _, _, _, importr = _load_rpy2()
    fwb = importr("fwildclusterboot")
    set_r_seeds(seed)

    formal_names = set(_probe_boottest_formals("boottest"))

    # Build basic argument dict; 'object' is the S3 dispatch object
    args = dict(
        param=str(param),
        B=int(B),
        impose_null=bool(impose_null),
    )

    if "type" in formal_names:
        args["type"] = str(type_)

    cl_names = _sanitize_clustid_names(df, clustid)
    if cl_names:
        if len(cl_names) == 1:
            cl_arg = cl_names[0]
        else:
            # multi-way clustering: character vector of names
            cl_arg = ro.StrVector(cl_names)

        if "clustid" in formal_names:
            args["clustid"] = cl_arg

    # First attempt with detected arguments
    try:
        return fwb.boottest(fit, **args)
    except Exception:
        # Retry without 'type' if that caused trouble
        args_retry = {k: v for k, v in args.items() if k != "type"}
        return fwb.boottest(fit, **args_retry)

def _probe_boottest_names(fn_name: str) -> set:
    ro, _, _, _, importr = _load_rpy2()
    fwb = importr("fwildclusterboot")
    f = getattr(fwb, fn_name)
    try:
        formals = list(ro.r("function(f) names(formals(f))")(f))
        return set(map(str, formals))
    except Exception:
        # conservative default
        return {"B","weights","type","clustid","cluster","impose_null","R","r","param"}


def mboottest_fixest(
    df: pd.DataFrame,
    fit: Any,                  # kept for API compatibility, unused
    joint_params: Optional[Sequence[str]],     # names of coefficients to test jointly
    B: int,
    impose_null: bool,
    clustid: Optional[Sequence[str]],
    type_: str,
) -> float:
    """
    Wrapper around fwildclusterboot::mboottest for a fixest (feols) object
    using the Julia backend (WildBootTests.jl).

    We follow the fwildclusterboot examples and use
    clubSandwich::constrain_zero() to construct the restriction matrix R
    for H0: R * beta = 0, where each row sets one coefficient in
    `joint_params` to zero.

    This avoids hand-rolling R and prevents the Julia error:

        "Null hypothesis or model constraints are inconsistent or redundant."

    Parameters
    ----------
    df : DataFrame
        Data used to fit the fixest model (subsample).
    fit : R object
        fixest::feols fitted model.
    param : str or None
        Unused here; present for signature compatibility.
    joint_params : list[str] or None
        Names of coefficients to test jointly equal to zero.
    B : int
        Number of bootstrap replications.
    impose_null : bool
        Whether to impose the null in the bootstrap DGP.
    clustid : list[str] or None
        Names of cluster variables (columns in df).
    type_ : str
        Bootstrap weight type (e.g. "rademacher").

    Returns
    -------
    float
        Wild cluster bootstrap p-value for H0: all `joint_params` = 0,
        or NaN on failure.
    """
    ro, _, _, _, importr = _load_rpy2()
    if not joint_params:
        # Nothing to test
        return float("nan")

    # Import R packages
    fwb = importr("fwildclusterboot")
    try:
        club = importr("clubSandwich")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "R package 'clubSandwich' is required for joint WCB tests "
            "(mboottest_fixest). Please install it in R via "
            "`install.packages('clubSandwich')`."
        ) from e

    # 1) Map coefficient NAMES -> POSITIONS
    coef_vec = ro.r["coef"](fit)
    coef_names = [str(n) for n in coef_vec.names]

    idx = []
    for name in joint_params:
        try:
            j = coef_names.index(str(name))
        except ValueError as exc:
            raise ValueError(
                f"[mboottest_fixest] Coefficient '{name}' not found in fixest "
                f"coefficients: {coef_names}"
            ) from exc
        # R is 1-based
        idx.append(j + 1)

    # 2) Let clubSandwich build a consistent R matrix:
    #    R <- clubSandwich::constrain_zero(idx, coef(fit))
    idx_vec = ro.IntVector(idx)
    R_mat = club.constrain_zero(idx_vec, coef_vec)

    # 3) Build clustid argument from df column names
    cl_names = _sanitize_clustid_names(df, clustid or [])
    if not cl_names:
        raise ValueError(
            "[mboottest_fixest] No valid cluster variable found. "
            f"Requested clustid={clustid}, df columns={list(df.columns)}"
        )
    if len(cl_names) == 1:
        cl_arg = cl_names[0]
    else:
        cl_arg = ro.StrVector(cl_names)

    # 4) Assemble mboottest arguments
    args: Dict[str, Any] = {
        "object": fit,
        "clustid": cl_arg,
        "B": int(B),
        "R": R_mat,
        "impose_null": bool(impose_null),
        "type": str(type_),
    }

    # NOTE: we let fwildclusterboot / WildBootTests.jl choose engine.
    # If you want to force Julia explicitly, you *can* add:
    #   args["engine"] = "WildBootTests.jl"
    # but it's usually not necessary.

    # 5) Call mboottest
    try:
        res = fwb.mboottest(**args)
    except ro.RRuntimeError as e:
        # If we hit "unused argument" errors for old versions, try stripping
        # type/clustid once and re-running.
        msg = str(e)
        if "unused argument" in msg:
            for bad in ("type", "weights", "clustid", "cluster"):
                args.pop(bad, None)
            res = fwb.mboottest(**args)
        else:
            raise

    # 6) Extract p-value via pval() S3 method
    try:
        p_fun = ro.r("pval")
        p_vec = p_fun(res)
        return float(p_vec[0])
    except Exception:
        # Fallback: try common slots
        for slot in ("p_val", "p.value", "pval", "p"):
            try:
                val = res.rx2(slot)
                if val is not None:
                    return float(val[0])
            except Exception:
                continue

    return float("nan")
