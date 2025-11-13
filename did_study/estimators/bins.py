# did_study/estimators/bins.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..helpers.config import StudyConfig
from ..helpers.utils import log_wcb_call, resolve_fe_terms
from ..robustness.wcb import FitSpec, TestSpec, WildClusterBootstrap
from ..robustness.stats.mde import analytic_mde_from_se

if TYPE_CHECKING:  # pragma: no cover
    from ..helpers.preparation import PanelData

Number = Union[int, float]


def _sanitize_bin_label(label: str) -> str:
    """Convert arbitrary bin labels into safe column-name fragments."""
    safe = re.sub(r"[^0-9A-Za-z]+", "_", label)
    safe = safe.strip("_")
    if not safe:
        safe = "bin"
    return safe


def _series_from_result(values: Any, model: Any, prefix: str) -> pd.Series:
    """Wrap statsmodels arrays into Series with sensible names."""
    if isinstance(values, pd.Series):
        return values
    names = getattr(model, "exog_names", None) or getattr(getattr(model, "data", None), "xnames", None)
    if names is None:
        names = [f"{prefix}{i}" for i in range(len(values))]
    return pd.Series(values, index=list(names))


# ======================================================================
# Helper: construct human-readable dose bins
# ======================================================================

def make_dose_bins(
    df: pd.DataFrame,
    dose_col: str,
    edges: Optional[Sequence[Number]] = None,
    quantiles: Optional[Sequence[float]] = None,
    include_untreated: bool = True,
    untreated_label: str = "untreated",
    right: bool = False,
    precision: int = 2,
) -> pd.Categorical:
    """
    Create a categorical dose-bin variable.

    This is the function used by PanelData._bin_dose_absorbing, which
    calls it like:

        make_dose_bins(
            g,
            dose_col="dose_level",
            edges=edges_used,
            quantiles=None,
            include_untreated=False,
            untreated_label="untreated",
            right=cfg.dose_bins_right_closed,
            precision=2,
        )

    Parameters
    ----------
    df :
        DataFrame containing the dose column.
    dose_col :
        Name of the column with the (continuous) dose variable.
    edges :
        Explicit bin edges (including lower and upper bounds).
        Example: [0.1, 0.5, 2.0, np.inf].
    quantiles :
        Alternatively, a sequence of quantiles in (0, 1) to be used as
        bin edges (on the *positive* part of the dose distribution).
        Only one of `edges` or `quantiles` should be supplied.
    include_untreated :
        If True, observations with non-positive (or missing) dose get
        their own “untreated” bin label. If False, they stay as NaN.
    untreated_label :
        Label used for the untreated bin.
    right :
        Whether the intervals are right-closed (like pandas.cut).
    precision :
        Number of decimals to display in bin labels.

    Returns
    -------
    pd.Categorical
        Categorical series of the same length as df.index, with
        ordered bin labels.
    """
    if edges is not None and quantiles is not None:
        raise ValueError("Specify either 'edges' or 'quantiles', not both.")

    dose = df[dose_col].astype(float)

    # Treated portion (strictly positive dose)
    mask_pos = dose > 0
    treated = dose[mask_pos]

    if edges is None and quantiles is None:
        raise ValueError("Either 'edges' or 'quantiles' must be provided.")

    if edges is None:
        # Use quantiles on the positive part
        qs = np.unique(quantiles)
        if qs[0] <= 0 or qs[-1] >= 1:
            raise ValueError("Quantiles should lie strictly between 0 and 1.")
        finite_vals = treated.replace([np.inf, -np.inf], np.nan).dropna()
        q_vals = finite_vals.quantile(qs).values
        q_vals = np.unique(q_vals)
        edges_arr = np.concatenate(([finite_vals.min()], q_vals, [finite_vals.max()]))
    else:
        edges_arr = np.asarray(edges, dtype=float)

    # Defensive: require at least two finite edges
    if len(edges_arr) < 2:
        raise ValueError("At least two bin edges are required.")

    # Build labels like "[0.1, 0.5)" or "(0.5, 2.0]"
    labels: List[str] = []
    for lo, hi in zip(edges_arr[:-1], edges_arr[1:]):
        if np.isinf(hi):
            hi_str = "inf"
        else:
            hi_str = f"{hi:.{precision}f}".rstrip("0").rstrip(".")

        if np.isinf(lo):
            lo_str = "-inf"
        else:
            lo_str = f"{lo:.{precision}f}".rstrip("0").rstrip(".")

        if right:
            lab = f"({lo_str}, {hi_str}]"
        else:
            lab = f"[{lo_str}, {hi_str})"
        labels.append(lab)

    # Cut treated observations
    treated_bins = pd.cut(
        treated,
        bins=edges_arr,
        labels=labels,
        right=right,
        include_lowest=True,
        ordered=True,
    )

    # Initialize all as NaN, then fill treated
    out = pd.Series(pd.Categorical([np.nan] * len(df), categories=labels, ordered=True),
                    index=df.index)

    out.loc[mask_pos] = treated_bins.astype("category")

    if include_untreated:
        # Add an explicit "untreated" category and assign to non-positive dose
        categories = [untreated_label] + labels
        out = out.cat.add_categories([untreated_label])  # type: ignore[arg-type]
        out.loc[~mask_pos] = untreated_label
        out = out.astype(pd.CategoricalDtype(categories=categories, ordered=True))

    return out.astype("category")


# ======================================================================
# Binned ATT estimation
# ======================================================================

@dataclass
class BinAttResult:
    """
    Result object for bin-level ATT estimation.
    """
    bin_names: List[str]
    coef: pd.Series
    se: pd.Series
    p_value: pd.Series
    n_obs: int
    n_treated_bins: pd.Series
    mde: Optional[pd.Series] = None
    joint_test_pvalue: Optional[float] = None
    p_value_wcb: Optional[pd.Series] = None
    cluster_counts: Optional[pd.Series] = None


def _bin_categorical(
    df: pd.DataFrame,
    bin_col: str,
    reference: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Turn a categorical bin variable into dummy regressors with safe column names.

    Returns
    -------
    (df_with_dummies, dummy_names, label_map)
      - dummy_names are safe column identifiers added to df.
      - label_map maps dummy_name -> original human-readable bin label.
    """
    c = df[bin_col].astype("category")
    categories = list(c.cat.categories)
    if not categories:
        return df.copy(), [], {}

    if reference is None:
        reference = categories[0]

    dummies = pd.get_dummies(c, prefix="", prefix_sep="", drop_first=False, dtype=float)
    if reference in dummies.columns:
        dummies = dummies.drop(columns=[reference])

    if dummies.empty:
        return df.copy(), [], {}

    df_out = pd.concat([df, dummies], axis=1)

    rename_map: Dict[Any, str] = {}
    label_map: Dict[str, str] = {}
    dummy_names: List[str] = []
    used_names: set[str] = set(str(col) for col in df.columns)

    for col in dummies.columns:
        label_str = str(col)
        base = f"trbin_{_sanitize_bin_label(label_str)}"
        name = base
        counter = 1
        while name in used_names:
            counter += 1
            name = f"{base}_{counter}"
        rename_map[col] = name
        label_map[name] = label_str
        dummy_names.append(name)
        used_names.add(name)

    if rename_map:
        df_out = df_out.rename(columns=rename_map)

    return df_out, dummy_names, label_map


def _build_bin_dummies(
    df: pd.DataFrame,
    bin_col: str,
    untreated_label: str = "untreated",
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Convenience wrapper that:
      1. Treats `untreated_label` as the reference bin.
      2. Adds dummy columns trbin_<label> for all other bins.
    """
    c = df[bin_col].astype("category")
    if len(c.cat.categories) == 0:
        return df.copy(), [], {}

    if untreated_label in c.cat.categories:
        ref = untreated_label
    else:
        ref = c.cat.categories[0]
    return _bin_categorical(df, bin_col=bin_col, reference=ref)


def _cluster_terms_from_args(
    df: pd.DataFrame,
    config: StudyConfig,
    wcb_args: Optional[Dict[str, Any]],
) -> List[str]:
    spec: Optional[Union[str, Sequence[str]]] = None
    if wcb_args:
        spec = (
            wcb_args.get("cluster_spec")
            or wcb_args.get("cluster_terms")
            or wcb_args.get("cluster")
        )
    if spec is None:
        spec = getattr(config, "cluster_col", None)
    terms: List[str] = []
    if isinstance(spec, str):
        spec = [spec]
    for term in spec or []:
        if term in df.columns and term not in terms:
            terms.append(term)
    if not terms and hasattr(config, "unit_cols"):
        for col in getattr(config, "unit_cols", ()):
            if col in df.columns:
                terms.append(col)
                break
    if not terms and "unit_id" in df.columns:
        terms = ["unit_id"]
    return terms


def estimate_binned_att_o(
    panel: PanelData,
    config: StudyConfig,
    fe_terms: Optional[Sequence[str]] = None,
    *,
    untreated_label: str = "untreated",
    wcb_args: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> BinAttResult:
    """
    Estimate bin-specific ATTs using a two-way FE regression with bin
    dummies, optionally using the WildClusterBootstrap wrapper to obtain
    p-values.

    This is deliberately written to be robust and *not* to depend on
    rpy2/Julia. The `WildClusterBootstrap` class in robustness.wcb uses
    statsmodels with cluster-robust covariance internally.

    Parameters
    ----------
    panel :
        Prepared PanelData output (contains df, covariates, outcome).
    config :
        Study configuration with defaults (e.g. unit/year columns).
    fe_terms :
        Optional override for fixed-effect columns.
    untreated_label :
        Label of the untreated bin (reference category).
    wcb_args :
        Optional WildClusterBootstrap configuration dict.
    seed :
        Optional seed forwarded to the bootstrap runner.

    Returns
    -------
    BinAttResult
    """
    df = panel.panel.copy()
    if df.empty or "dose_bin" not in df.columns:
        empty = pd.Series([], dtype=float)
        return BinAttResult(
            bin_names=[],
            coef=empty,
            se=empty,
            p_value=empty,
            n_obs=int(df.shape[0]),
            n_treated_bins=pd.Series([], dtype=float),
            mde=None,
            joint_test_pvalue=None,
            p_value_wcb=None,
            cluster_counts=None,
        )

    outcome = panel.outcome_name or getattr(config, "outcome_col", "outcome")
    bin_col = "dose_bin"
    controls = list(panel.info.get("covariates_used", []) or [])

    cluster_terms = _cluster_terms_from_args(df, config, wcb_args)
    cluster_col = cluster_terms[0] if cluster_terms else None

    year_col = getattr(config, "year_col", "Year")
    unit_for_fe: Optional[str] = cluster_col
    if not unit_for_fe:
        cfg_unit_cols = getattr(config, "unit_cols", None)
        if cfg_unit_cols:
            unit_for_fe = cfg_unit_cols[0]
    if not unit_for_fe:
        unit_for_fe = "unit_id"

    outcome_lower = str(outcome).lower()
    include_unit_fe_default = not outcome_lower.startswith("d_")
    default_fe: List[str] = [year_col]
    if include_unit_fe_default:
        if cluster_col:
            default_fe.insert(0, cluster_col)
        elif "unit_id" in df.columns:
            default_fe.insert(0, "unit_id")
    fe_terms = list(fe_terms) if fe_terms else default_fe

    use_wcb = bool(wcb_args)

    # ------------------------------------------------------------------
    # 1. Build bin dummies
    # ------------------------------------------------------------------
    df_work, bin_dummies, bin_label_map = _build_bin_dummies(
        df.copy(), bin_col=bin_col, untreated_label=untreated_label
    )
    bin_labels = [bin_label_map.get(name, name) for name in bin_dummies]

    # Safety: no bins -> nothing to do
    if not bin_dummies:
        empty = pd.Series([], dtype=float)
        return BinAttResult(
            bin_names=[],
            coef=empty,
            se=empty,
            p_value=empty,
            n_obs=int(df_work.shape[0]),
            n_treated_bins=pd.Series([], dtype=float),
            mde=None,
            joint_test_pvalue=None,
            p_value_wcb=None,
            cluster_counts=None,
        )

    # ------------------------------------------------------------------
    # 2. OLS with fixed effects via statsmodels
    # ------------------------------------------------------------------
    rhs_terms: List[str] = []
    rhs_terms.extend(bin_dummies)
    rhs_terms.extend(controls)

    # Fixed effects as categorical dummies
    fe_terms = resolve_fe_terms(
        fe_terms,
        year_col=year_col,
        unit_col=unit_for_fe,
    )  # e.g. ["C(unit_id)", "C(Year)"]
    rhs_terms.extend(fe_terms)

    formula = f"{outcome} ~ " + " + ".join(rhs_terms)
    model = smf.ols(formula=formula, data=df_work)
    ols_res = model.fit()

    # ------------------------------------------------------------------
    # 3. Cluster-robust covariance
    # ------------------------------------------------------------------
    if cluster_col:
        robust_res = ols_res.get_robustcov_results(
            cov_type="cluster", groups=df_work[cluster_col]
        )
    else:
        robust_res = ols_res

    params = _series_from_result(robust_res.params, robust_res.model, "param_").reindex(bin_dummies)
    ses = _series_from_result(robust_res.bse, robust_res.model, "se_").reindex(bin_dummies)

    # ------------------------------------------------------------------
    # 4. p-values via WildClusterBootstrap wrapper (or analytic)
    # ------------------------------------------------------------------
    try:
        ttest = robust_res.t_test(bin_dummies)
        analytic_vals = np.asarray(np.atleast_1d(ttest.pvalue)).reshape(-1)
        analytic_pvals = pd.Series(analytic_vals, index=bin_dummies, dtype=float)
    except Exception:
        analytic_pvals = pd.Series([float("nan")] * len(bin_dummies), index=bin_dummies, dtype=float)

    wcb_runner: Optional[WildClusterBootstrap] = None
    pvals_wcb: Optional[pd.Series] = None
    if wcb_args:
        fit_spec = FitSpec(
            outcome=outcome,
            regressors=list(bin_dummies) + list(controls),
            fe=list(fe_terms),
            cluster=list(cluster_terms),
        )
        wcb_runner = WildClusterBootstrap(
            df=df_work,
            fit_spec=fit_spec,
            B=wcb_args['B'],
            weights=wcb_args['weights'],
            seed=seed,
            impose_null=wcb_args['impose_null'],
        )

        wcb_values: List[float] = []
        for param in bin_dummies:
            pval = wcb_runner.pvalue(TestSpec(param=param))
            wcb_values.append(pval)
        pvals_wcb = pd.Series(wcb_values, index=bin_dummies, dtype=float)

    # ------------------------------------------------------------------
    # 5. MDEs (analytic, using cluster-robust SEs)
    # ------------------------------------------------------------------
    n_clusters_mde: Optional[int] = None
    if cluster_col and cluster_col in df_work.columns:
        n_clusters_mde = int(df_work[cluster_col].nunique())
    elif "unit_id" in df_work.columns:
        n_clusters_mde = int(df_work["unit_id"].nunique())

    if n_clusters_mde and n_clusters_mde > 0:
        mde = ses.apply(
            lambda x: analytic_mde_from_se(float(x), n_clusters_mde)
            if pd.notna(x)
            else np.nan
        )
    else:
        mde = None

    # ------------------------------------------------------------------
    # 6. Optional joint test across *all* treated bins
    # ------------------------------------------------------------------
    joint_pvalue: Optional[float] = None
    try:
        if use_wcb and len(bin_dummies) > 1 and wcb_runner is not None:
            joint_pvalue = wcb_runner.pvalue(TestSpec(joint_zero=list(bin_dummies)))
    except Exception:
        # Fallback: analytic Wald test with cluster-robust cov
        try:
            constraints = " = 0, ".join(bin_dummies) + " = 0"
            wald_res = robust_res.wald_test(constraints)
            joint_pvalue = float(wald_res.pvalue)
        except Exception:
            joint_pvalue = None

    # Count treated observations per bin
    n_treated_bins = df_work.groupby(bin_col)[outcome].count()
    n_treated_bins = n_treated_bins.reindex(bin_labels)
    cluster_counts: Optional[pd.Series] = None
    if cluster_col and cluster_col in df_work.columns:
        cluster_counts = df_work.groupby(bin_col)[cluster_col].nunique().reindex(bin_labels)

    return BinAttResult(
        bin_names=bin_labels,
        coef=params,
        se=ses,
        p_value=analytic_pvals,
        n_obs=int(df_work.shape[0]),
        n_treated_bins=n_treated_bins,
        mde=mde,
        joint_test_pvalue=joint_pvalue,
        p_value_wcb=pvals_wcb,
        cluster_counts=cluster_counts,
    )
