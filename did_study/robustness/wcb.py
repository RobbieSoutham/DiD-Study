# did_study/robustness/wcb.py
from __future__ import annotations

from typing import Optional, Sequence, List
import numpy as np
import pandas as pd

# rpy2 conversions
from rpy2.robjects import default_converter, pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# ---------------------------
# helpers
# ---------------------------

def _set_seeds_r(seed: Optional[int]):
    if seed is None:
        return
    try:
        ro.r(f"set.seed({int(seed)})")
    except Exception:
        pass
    try:
        ro.r("if (requireNamespace('dqrng', quietly=TRUE)) dqrng::dqset.seed(%d)" % int(seed))
    except Exception:
        pass


def _sanitize_fe_list(fe: Sequence[str]) -> list[str]:
    out: list[str] = []
    for x in fe or []:
        xs = str(x).replace("|", "").strip()
        if xs and xs not in out:
            out.append(xs)
    return out


def _parse_cluster_vars(cluster: str | Sequence[str]) -> List[str]:
    """
    Accepts:
      - "unit_id"
      - "Year + unit_id"
      - "~ Year + unit_id"
      - ["Year", "unit_id"]
    Returns a clean list of variable names.
    """
    if cluster is None:
        return []
    if isinstance(cluster, (list, tuple)):
        return [str(c).strip() for c in cluster if str(c).strip()]
    s = str(cluster).strip()
    if s.startswith("~"):
        s = s[1:].strip()
    parts = [p.strip() for p in s.split("+")]
    return [p for p in parts if p]


def _sanitize_varlist(df: pd.DataFrame, items: Sequence) -> list[str]:
    """
    Coerce any mixture of strings/Series/accidentally-passed values
    into a clean list of *column names that exist in df*.
    - If an element is a pandas Series, use its .name
    - If a string contains '+', split and keep parts that are columns
    - Drop anything not in df.columns
    """
    cols = set(df.columns)
    out: list[str] = []
    for it in (items or []):
        name: str | None = None
        if isinstance(it, pd.Series):
            name = str(it.name) if it.name is not None else None
        else:
            s = str(it).strip()
            if not s:
                name = None
            elif "+" in s:
                for p in [pp.strip() for pp in s.split("+") if pp.strip()]:
                    if p in cols and p not in out:
                        out.append(p)
                name = None  # handled via split
            else:
                name = s

        if name and (name in cols) and (name not in out):
            out.append(name)
    return out


def _build_feols_formula(outcome: str, regressors: Sequence[str], fe: Sequence[str]) -> str:
    # fixest syntax:  y ~ x1 + x2 | fe1 + fe2
    fe = _sanitize_fe_list(fe)
    rhs = " + ".join(map(str, regressors)) if regressors else "1"
    fe_rhs = " + ".join(map(str, fe)) if fe else ""
    fe_part = f" | {fe_rhs}" if fe_rhs else ""
    return f"{outcome} ~ {rhs}{fe_part}"


def _subset_for_r(
    df: pd.DataFrame,
    outcome: str,
    regressors: Sequence[str],
    fe: Sequence[str],
    cluster: str | Sequence[str],
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    Returns (sub_df, used_regressors, used_fe, used_cluster_vars)
    All lists are sanitized to refer to columns that exist in df.
    """
    used_reg = _sanitize_varlist(df, regressors)
    used_fe  = _sanitize_varlist(df, fe)
    cl_vars  = _parse_cluster_vars(cluster)

    keep = set([outcome]) | set(used_reg) | set(used_fe) | set(cl_vars)
    cols = [c for c in keep if c in df.columns]
    sub = df.loc[:, cols].copy()

    # Drop columns that cause rpy2 grief and aren’t needed in R:
    sub = sub.drop(columns=["dose_bin", "event_time"], errors="ignore")

    # Ensure regressors are numeric (controls sometimes arrive as object)
    for c in used_reg:
        if c in sub.columns and not np.issubdtype(sub[c].dtype, np.number):
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

    return sub, used_reg, used_fe, cl_vars


def _boottest_kw(joint: bool) -> dict:
    """
    Version-proof argument discovery for fwildclusterboot.

    For boottest (single-coef), probe the generic.
    For mboottest (joint), probe both the generic and the fixest S3 method.
    """
    fn_name = "mboottest" if joint else "boottest"

    names_generic = []
    try:
        names_generic = list(ro.r(f"names(formals(fwildclusterboot::{fn_name}))"))
    except Exception:
        pass

    names_all = set(names_generic)

    if joint:
        try:
            names_fixest = list(ro.r(f"names(formals(getS3method('{fn_name}', 'fixest')))"))
            names_all |= set(names_fixest)
        except Exception:
            pass

    return dict(
        use_type=("type" in names_all),
        use_weights=("weights" in names_all),
        use_clustid=("clustid" in names_all),
        use_cluster=("cluster" in names_all),
    )


def _extract_pval(res) -> float:
    import numpy as _np
    try:
        for key in ("p_val", "p.value", "p"):
            try:
                arr = _np.array(res.rx2(key))
                if arr.size > 0 and _np.isfinite(arr[0]):
                    return float(arr[0])
            except Exception:
                pass
    except Exception:
        pass
    return float("nan")


def _clustid_arg_r(df_py: pd.DataFrame, cluster_vars: list[str]):
    """
    Build a robust 'clustid' argument for fwildclusterboot:
      - one-way: return R vector
      - multi-way: return R data.frame
    """
    if not cluster_vars:
        return None
    if len(cluster_vars) == 1:
        col = cluster_vars[0]
        with localconverter(default_converter + pandas2ri.converter):
            r_vec = ro.conversion.py2rpy(df_py[col])
        return r_vec
    else:
        with localconverter(default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df_py[cluster_vars].copy())
        return r_df


# ---------------------------
# Public API
# ---------------------------

def wcb_att_pvalue_r(
    df: pd.DataFrame,
    *,
    outcome: str,
    regressors: Sequence[str],
    fe: Sequence[str],
    cluster: str,
    param: str,
    B: int = 9999,
    weights: str = "webb",   # "rademacher" | "webb" | ...
    impose_null: bool = True,
    seed: Optional[int] = None,
) -> float:
    """
    Wild-cluster bootstrap p for a SINGLE coefficient (pooled ATT^o).
    Uses fwildclusterboot::boottest with fixest::feols.
    """
    try:
        fixest = importr("fixest")
        fwb = importr("fwildclusterboot")
        _set_seeds_r(seed)

        sub, used_reg, used_fe, cl_vars = _subset_for_r(df, outcome, regressors, fe, cluster)
        with localconverter(default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(sub)

        fml_str = _build_feols_formula(outcome, used_reg, used_fe)
        fml = ro.r(f"as.formula('{fml_str}')")
        model = fixest.feols(fml, data=r_df)

        # Keep your original formula-based clustering for boottest()
        cl = ro.r(f"~ {' + '.join(cl_vars)}") if cl_vars else ro.r("~ 1")

        kw = _boottest_kw(joint=False)
        args = dict(B=int(B), impose_null=bool(impose_null))
        if kw["use_type"]:
            args["type"] = str(weights)
        elif kw["use_weights"]:
            args["weights"] = str(weights)
        if kw["use_clustid"]:
            args["clustid"] = cl
        elif kw["use_cluster"]:
            args["cluster"] = cl

        res = fwb.boottest(model, param=str(param), **args)
        return _extract_pval(res)

    except Exception as e:
        print(f"[WCB/R] boottest failed: {e}")
        return float("nan")


def wcb_joint_pvalue_r(
    df,
    *,
    outcome: str,
    regressors,
    fe,
    cluster: str,
    R=None,
    r=None,
    joint_zero=(),
    B: int = 9999,
    weights: str = "webb",
    impose_null: bool = True,
    seed=None,
) -> float:
    """
    Wild-cluster bootstrap p for a JOINT test (omnibus / equality).
    Fixes:
      - Sanitize regressors/FE into column names (prevents str2lang(...) on values).
      - Build 'clustid' from actual data (vector/data.frame) instead of a formula.
      - Probe the fixest S3 method and prefer clustid= when available.
      - Ensure 'r' is a true 1-D numeric vector.
    """
    try:
        fixest = importr("fixest")
        fwb = importr("fwildclusterboot")
        _set_seeds_r(seed)

        sub, used_reg, used_fe, cl_vars = _subset_for_r(df, outcome, regressors, fe, cluster)
        cl_arg = _clustid_arg_r(sub, cl_vars)

        with localconverter(default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(sub)

        fml_str = _build_feols_formula(outcome, used_reg, used_fe)
        fml = ro.r(f"as.formula('{fml_str}')")
        model = fixest.feols(fml, data=r_df)

        # ---- Build R and r properly ----
        if R is None or r is None:
            cols = list(ro.r("function(m) colnames(stats::model.matrix(m))")(model))
            if not joint_zero:
                raise ValueError("Provide (R, r) or joint_zero names.")
            R_np = np.zeros((len(joint_zero), len(cols)), dtype=float)
            for i, name in enumerate(joint_zero):
                if name not in cols:
                    raise ValueError(f"Joint test param not in model: {name}")
                R_np[i, cols.index(name)] = 1.0
            r_np = np.zeros((len(joint_zero),), dtype=float)
        else:
            R_np = np.asarray(R, dtype=float)
            r_np = np.asarray(r, dtype=float).reshape(-1)   # ensure 1D

        # Convert to R types
        R_vec = ro.FloatVector(R_np.ravel(order="C").tolist())
        R_r   = ro.r["matrix"](R_vec, nrow=R_np.shape[0], ncol=R_np.shape[1], byrow=True)
        r_vec = ro.FloatVector(r_np.tolist())

        # Discover args (generic + fixest S3) and prefer clustid
        kw = _boottest_kw(joint=True)

        args = dict(B=int(B), impose_null=bool(impose_null), R=R_r, r=r_vec)
        if kw["use_type"]:
            args["type"] = str(weights)
        elif kw["use_weights"]:
            args["weights"] = str(weights)

        if kw["use_clustid"]:
            args["clustid"] = cl_arg    # pass vector/data.frame
        elif kw["use_cluster"]:
            args["cluster"]  = cl_arg

        res = fwb.mboottest(model, **args)
        try:
            return float(np.array(res.rx2("p_val"))[0])
        except Exception:
            return float("nan")

    except Exception as e:
        print(f"[WCB/R] mboottest failed: {e}")
        return float("nan")
