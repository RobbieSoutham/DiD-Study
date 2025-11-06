# r_interface.py
"""
R interop for Wild Cluster Bootstrap (fwildclusterboot) using fixest::feols.

Public API
----------
- wcb_att_pvalue_r
- wcb_joint_pvalue_r
- wcb_equal_bins_pvalue_r

Compat helpers
--------------
- to_r_ready_df(df, needed_cols=None, year_col=None, cluster_col=None) -> (clean_df, name_map)
- py_df_to_r(df)
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Dict
import re
import numpy as np
import pandas as pd

from rpy2.robjects import default_converter, pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import Formula

# R pkgs
base = importr("base")
stats = importr("stats")
fixest = importr("fixest")
fwcb = importr("fwildclusterboot")

_SAFE_RE = re.compile(r"[^A-Za-z0-9_]")


def _sanitize(name: str) -> str:
    s = _SAFE_RE.sub("_", str(name))
    if not s or s[0].isdigit():
        s = "v_" + s
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "v"


def _make_name_map(cols: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    used: set[str] = set()
    for c in cols:
        s = _sanitize(c)
        base_s = s; k = 1
        while s in used:
            k += 1; s = f"{base_s}_{k}"
        used.add(s)
        out[c] = s
    return out


def _apply_name_map(df: pd.DataFrame, name_map: Dict[str, str]) -> pd.DataFrame:
    renamer = {c: name_map[c] for c in df.columns if c in name_map}
    return df.rename(columns=renamer)


def _map_list(names: Sequence[str], name_map: Dict[str, str]) -> List[str]:
    return [name_map.get(n, _sanitize(n)) for n in names]


def py_df_to_r(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("py_df_to_r expects a pandas DataFrame")
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


def to_r_ready_df(
    df: pd.DataFrame,
    *,
    needed_cols: Optional[Sequence[str]] = None,
    year_col: Optional[str] = None,
    cluster_col: Optional[str] = None,
    drop_problematic: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Sanitize a pandas DataFrame and return (clean_df, name_map).
    - Subset to needed_cols if provided.
    - Drop Interval/NA-heavy columns (notably 'dose_bin').
    - Ensure trbin_* columns are numeric (float).
    - Coerce booleans to ints.
    - Sanitize column names for R and return the mapping.
    """
    out = df.copy()

    if needed_cols:
        keep = [c for c in needed_cols if c in out.columns]
        out = out.loc[:, keep].copy()

    if drop_problematic:
        for c in list(out.columns):
            if str(out[c].dtype).startswith("interval") or c == "dose_bin":
                out = out.drop(columns=[c])

    for cname in ("dose_bin_label", "dose_bin_str"):
        if cname in out.columns:
            out[cname] = out[cname].astype("string")

    for c in out.columns:
        if c.startswith("trbin_"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        if out[c].dtype == "bool":
            out[c] = out[c].astype("int64")

    name_map = _make_name_map(list(out.columns))
    out = _apply_name_map(out, name_map)
    return out, name_map


def _clust_formula(cluster: Optional[str]):
    if cluster is None or str(cluster).strip() == "":
        return ro.NULL
    return Formula("~ " + str(cluster))


def _fit_feols(df: pd.DataFrame, y: str, X: Sequence[str], FE: Sequence[str], seed: Optional[int] = None):
    if seed is not None:
        base.set_seed(int(seed))
        try:
            dqrng = importr("dqrng")
            dqrng.dqset_seed(int(seed))
        except Exception:
            pass

    y_f = str(y)
    X_f = " + ".join(X) if X else "1"
    FE_f = " + ".join(FE)
    fml = Formula(f"{y_f} ~ {X_f}" + (f" | {FE_f}" if FE_f else ""))
    r_df = py_df_to_r(df)
    fit = fixest.feols(fml, data=r_df)
    return fit


def _boottest_single(fit, param: str, clustid, B: int, weights: str, impose_null: bool, seed: Optional[int] = None) -> float:
    args = {"param": StrVector([param]), "B": int(B), "weights": str(weights)}
    if clustid is not None:
        args["clustid"] = clustid
    if impose_null:
        args["R"] = ro.NULL; args["r"] = ro.NULL
    if seed is not None:
        base.set_seed(int(seed))
        try:
            dqrng = importr("dqrng")
            dqrng.dqset_seed(int(seed))
        except Exception:
            pass
    res = fwcb.mboottest(fit, **args)
    p = None
    for key in ("p_val", "p.value", "p"):
        try:
            v = np.array(res.rx2(key))
            if v.size and np.isfinite(v[0]):
                p = float(v[0]); break
        except Exception:
            pass
    return float("nan") if p is None else float(p)


def _boottest_joint_zero(fit, params: Sequence[str], clustid, B: int, weights: str, impose_null: bool, seed: Optional[int] = None) -> float:
    args = {"param": StrVector(list(params)), "B": int(B), "weights": str(weights)}
    if clustid is not None:
        args["clustid"] = clustid
    if not impose_null:
        args["R"] = ro.NULL; args["r"] = ro.NULL
    if seed is not None:
        base.set_seed(int(seed))
        try:
            dqrng = importr("dqrng")
            dqrng.dqset_seed(int(seed))
        except Exception:
            pass
    res = fwcb.mboottest(fit, **args)
    p = None
    for key in ("p_val", "p.value", "p"):
        try:
            v = np.array(res.rx2(key))
            if v.size and np.isfinite(v[0]):
                p = float(v[0]); break
        except Exception:
            pass
    return float("nan") if p is None else float(p)


def wcb_att_pvalue_r(
    df: pd.DataFrame,
    *,
    outcome: str,
    regressors: Sequence[str],
    fe: Sequence[str],
    cluster: Optional[str],
    param: str,
    B: int = 9999,
    weights: str = "webb",
    impose_null: bool = True,
    seed: Optional[int] = None,
) -> float:
    needed = [outcome] + list(regressors) + list(fe) + ([cluster] if cluster else [])
    clean_df, name_map = to_r_ready_df(df, needed_cols=needed, year_col=(fe[0] if fe else None), cluster_col=cluster)
    y_s   = name_map.get(outcome, _sanitize(outcome))
    X_s   = [name_map.get(x, _sanitize(x)) for x in regressors]
    FE_s  = [name_map.get(f, _sanitize(f)) for f in fe]
    clu_s = name_map.get(cluster, cluster) if cluster else None
    par_s = name_map.get(param, _sanitize(param))
    fit = _fit_feols(clean_df, y_s, X_s, FE_s, seed)
    clustid = _clust_formula(clu_s)
    return _boottest_single(fit, param=par_s, clustid=clustid, B=B, weights=weights, impose_null=impose_null, seed=seed)


def wcb_joint_pvalue_r(
    df: pd.DataFrame,
    *,
    outcome: str,
    regressors: Sequence[str],
    fe: Sequence[str],
    cluster: Optional[str],
    joint_zero: Sequence[str],
    B: int = 9999,
    weights: str = "webb",
    impose_null: bool = True,
    seed: Optional[int] = None,
) -> float:
    needed = [outcome] + list(regressors) + list(fe) + ([cluster] if cluster else [])
    clean_df, name_map = to_r_ready_df(df, needed_cols=needed, year_col=(fe[0] if fe else None), cluster_col=cluster)
    y_s   = name_map.get(outcome, _sanitize(outcome))
    X_s   = [name_map.get(x, _sanitize(x)) for x in regressors]
    FE_s  = [name_map.get(f, _sanitize(f)) for f in fe]
    clu_s = name_map.get(cluster, cluster) if cluster else None

    fit = _fit_feols(clean_df, y_s, X_s, FE_s, seed)
    clustid = _clust_formula(clu_s)
    params = [name_map.get(p, _sanitize(p)) for p in joint_zero]
    return _boottest_joint_zero(fit, params=params, clustid=clustid, B=B, weights=weights, impose_null=impose_null, seed=seed)


def wcb_equal_bins_pvalue_r(
    df: pd.DataFrame,
    *,
    outcome: str,
    regressors: Sequence[str],
    fe: Sequence[str],
    cluster: Optional[str],
    bin_params: Sequence[str],
    B: int = 9999,
    weights: str = "webb",
    impose_null: bool = True,
    seed: Optional[int] = None,
) -> float:
    """
    Test H0: all bin effects are equal by reparameterising as differences
    against the first bin column in bin_params.
    """
    if not bin_params:
        return float("nan")
    ref = bin_params[0]

    g = df.copy()
    for b in bin_params[1:]:
        g[f"diff__{b}__{ref}"] = g[b] - g[ref]

    diff_cols = [f"diff__{b}__{ref}" for b in bin_params[1:]]

    needed = [outcome] + list(regressors) + list(fe) + ([cluster] if cluster else []) + diff_cols
    clean_df, name_map = to_r_ready_df(g, needed_cols=needed, year_col=(fe[0] if fe else None), cluster_col=cluster)
    y_s   = name_map.get(outcome, _sanitize(outcome))
    X_base = [c for c in regressors if c not in bin_params]
    X_trans = [name_map.get(x, _sanitize(x)) for x in X_base] + [name_map[c] for c in diff_cols]
    FE_s  = [name_map.get(f, _sanitize(f)) for f in fe]
    clu_s = name_map.get(cluster, cluster) if cluster else None

    fit = _fit_feols(clean_df, y_s, X_trans, FE_s, seed)
    clustid = _clust_formula(clu_s)
    return _boottest_joint_zero(fit, params=[name_map[c] for c in diff_cols], clustid=clustid,
                                B=B, weights=weights, impose_null=impose_null, seed=seed)
