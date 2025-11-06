"""
Bins and binned ATT^o estimation utilities.

This module provides helpers to build economically meaningful dose bins and
estimate bin-specific ATT^o in a single, collapsed two‑way FE regression:

    y_it = sum_{b in B+} beta_b * 1{dose_bin_i,t = b} + X'_{it} gamma
            + alpha_i + tau_t + e_it,

where the baseline category is the "untreated" (or zero‑dose) bin. Coefficients
beta_b are interpreted as average treatment effects at bin b relative to
untreated, controlling for unit and time fixed effects (alpha_i, tau_t).

Key implementation choices
--------------------------
1) We force `dose_bin` to be an ordered *string* categorical (NOT pandas.Interval)
   to avoid rpy2 / design-matrix edge cases and collinearity issues stemming from
   auto-generated variable names. We also sanitize dummy names.

2) By default we estimate via statsmodels OLS with two‑way FE absorbed using
   categorical dummies (C(unit), C(year)) and cluster‑robust SEs at `cluster_col`.
   If the `linearmodels` package is available, we use PanelOLS with entity & time
   effects for speed and numerical stability.

3) Optional: if a local `.wcb` module is available (your project’s wrapper), we
   will also compute Wild Cluster Bootstrap (WCB) p‑values for each bin.

Returns a tidy table suitable for printing and plotting.

Author: your_name_here
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Try optional dependencies
try:  # linearmodels for fast FE
    from linearmodels.panel import PanelOLS  # type: ignore
    _HAVE_LM = True
except Exception:  # pragma: no cover - optional
    _HAVE_LM = False

import statsmodels.formula.api as smf
import statsmodels.api as sm

from ..helpers.config import StudyConfig
from ..robustness.stats.mde import analytic_mde_from_se

# Optional WCB bridge (kept very light-weight and *truly* optional)
try:
    from ..robustness.wcb import wcb_att_pvalue_r  # type: ignore
    _HAVE_WCB = True
except Exception:  # pragma: no cover - optional
    _HAVE_WCB = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_interval_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_interval_dtype(s)


def _sanitize_bin_label(x: Any) -> str:
    """Convert any bin label to a safe short token for column names.

    Examples
    --------
    "[0.5, 1.0)" -> "b_0p5_1p0"
    "(1, 5]"      -> "b_1_5"
    0             -> "b0"
    None          -> "untreated"
    "untreated"   -> "untreated"
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "untreated"
    s = str(x).strip()
    s = s.replace("∞", "inf").replace("-∞", "-inf")
    # Replace punctuation with underscores
    for ch in ["[", "]", "(", ")", ",", ":", ";", " ", "-", "+", "/", "\\", "|"]:
        s = s.replace(ch, "_")
    s = s.replace(".", "p")
    # Collapse multiple underscores
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_")
    if not s:
        s = "untreated"
    # Ensure starts with letter
    if not s[0].isalpha():
        s = "b_" + s
    return s


def _as_ordered_categorical(series: pd.Series, untreated_label: str = "untreated") -> pd.Categorical:
    """Ensure an ordered categorical with `untreated_label` as the first category.

    Accepts strings, numbers, or pandas.Interval. Converts all to strings.
    """
    if _is_interval_dtype(series):
        # Convert Interval to string labels to avoid rpy2/design-matrix issues
        values = series.astype(str)
    else:
        values = series.astype(object).where(series.notna(), other=untreated_label).astype(str)

    # If any value equals "nan" (from astype), coerce to untreated
    values = values.replace({"nan": untreated_label})

    cats = pd.Index(pd.unique(values))
    # Put untreated first if present
    if untreated_label in cats:
        ordered = [untreated_label] + [c for c in cats if c != untreated_label]
    else:
        ordered = list(cats)
    return pd.Categorical(values, categories=ordered, ordered=True)



def _bin_categorical(
    series: pd.Series | pd.DataFrame | Iterable,
    *,
    untreated_label: str = "untreated",
    order: Optional[List[str]] = None,
) -> pd.Categorical:
    """
    Convert a 'dose_bin'-like input into an **ordered** string categorical.
    Accepts Series, single-col DataFrame, numpy array, or list.
    NA -> untreated_label. Works for Interval dtype (stringifies cleanly).
    """
    s = _ensure_series(series)
    obj = s.astype(object).where(s.notna(), other=untreated_label)
    values = obj.astype(str)  # Intervals stringify fine here

    cats_obs = pd.Index(pd.unique(values))

    if order:
        ordered = [c for c in order if c in set(cats_obs)]
        cats = pd.Index(ordered) if ordered else cats_obs
    else:
        cats = cats_obs

    return pd.Categorical(values, categories=cats, ordered=True)


# ---------------------------------------------------------------------------
# Public helpers to create bins
# ---------------------------------------------------------------------------

def make_dose_bins(
    df: pd.DataFrame,
    *,
    dose_col: str,
    edges: Optional[Iterable[float]] = None,
    quantiles: Optional[Iterable[float]] = None,
    include_untreated: bool = True,
    untreated_label: str = "untreated",
    right: bool = False,
    precision: int = 2,
    label_fmt: str = "[{left}, {right})",
) -> pd.Series:
    """Create dose bins as an ordered *string* categorical.

    Either provide explicit `edges` (including min/max) or `quantiles`.

    - If `include_untreated` is True, rows with dose <= 0 (or NaN) get the
      `untreated_label` and are excluded from positive binning.
    - For quantiles, we compute on strictly positive doses only.

    Returns a pd.Categorical of labels with `untreated_label` first.
    """
    s = df[dose_col].copy()
    s_pos = s.where(s > 0, other=np.nan)

    if edges is not None and quantiles is not None:
        raise ValueError("Provide either `edges` or `quantiles`, not both.")

    pos_labels: pd.Series
    if edges is not None:
        edges = list(edges)
        if any(np.isnan(edges)):
            raise ValueError("`edges` contains NaNs.")
        if sorted(edges) != list(edges):
            raise ValueError("`edges` must be sorted.")
        # Apply to positive doses only
        pos_labels = pd.cut(s_pos, bins=edges, right=right, include_lowest=True)
    elif quantiles is not None:
        q = sorted(set(quantiles))
        # Compute quantiles on positive values only
        pos_vals = s_pos.dropna()
        if len(pos_vals) == 0:
            pos_labels = pd.Series(pd.Categorical([np.nan] * len(s)), index=s.index)
        else:
            qs = np.unique(np.clip(q, 0.0, 1.0))
            cuts = list(np.unique(np.quantile(pos_vals, qs)))
            cuts = sorted(set(cuts))
            if len(cuts) <= 1:
                pos_labels = pd.Series(pd.Categorical([np.nan] * len(s)), index=s.index)
            else:
                pos_labels = pd.cut(s_pos, bins=cuts, right=right, include_lowest=True)
    else:
        raise ValueError("You must provide `edges` or `quantiles`.")

    # Build human-readable labels like "[0.00, 1.00)"
    def _lab(iv: Any) -> Optional[str]:
        if pd.isna(iv):
            return None
        if isinstance(iv, pd.Interval):
            left = round(float(iv.left), precision)
            right_ = round(float(iv.right), precision)
            br_l = "(" if iv.open_left else "["
            br_r = ")" if iv.open_right else "]"
            return label_fmt.format(left=left, right=right_).replace("[", br_l).replace("]", br_r)
        return str(iv)

    labels_str = pos_labels.astype(object).map(_lab)

    if include_untreated:
        lbl = labels_str.where((s_pos > 0) & labels_str.notna(), other=untreated_label)
    else:
        lbl = labels_str

    return _as_ordered_categorical(lbl, untreated_label=untreated_label)


# ---------------------------------------------------------------------------
# Estimation: bin-specific ATT^o
# ---------------------------------------------------------------------------

@dataclass
class BinAttResult:
    table: pd.DataFrame
    model: Any
    used: pd.DataFrame
    name_map: Dict[str, str]
    baseline: str = "untreated"


def estimate_binned_att_o(
    panel: PanelData,
    config: StudyConfig,
) -> BinAttResult:
    """Estimate bin-specific ATT^o via a collapsed two‑way FE regression.

    Parameters
    ----------
    panel : PanelData
        Prepared panel with outcome, dose_bin, covariates and FE identifiers.
    config : StudyConfig
        Includes FE flags, cluster id, bootstrap controls and seed.

    Returns
    -------
    BinAttResult
        - table: rows for each positive bin with coef, se, p, ci_low, ci_high, N, G
        - model: fitted model object
        - used: the DataFrame slice used in estimation (post dropna, etc.)
        - name_map: mapping original bin label -> dummy column name

    Notes
    -----
    * The baseline is taken to be the first category of `dose_bin` after coercion,
      which we construct so that "untreated" is first when present.
    * We construct dummy names in a sanitized, short form to prevent collinearity
      and rpy2 conversion issues.
    """
    df = panel.panel.copy()
    if df.empty:
        raise ValueError("Panel is empty; cannot estimate binned ATT.")
    
    # Extract configuration values
    outcome_col = panel.outcome_name
    dose_bin_col = "dose_bin"
    year_col = config.year_col
    cluster_col = "unit_id"
    controls = panel.info.get("covariates_used", []) or []
    use_wcb = bool(getattr(config, "use_wcb", False))
    wcb_B = int(getattr(config, "wcb_B", 9999))
    wcb_weights = getattr(config, "wcb_weights", None)
    impose_null = bool(getattr(config, "impose_null", True))
    use_linearmodels = getattr(config, "use_linearmodels", None)

    # Guardrails
    needed = {outcome_col, dose_bin_col, year_col, cluster_col}
    missing = needed.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Clean outcome & key columns
    df = df.dropna(subset=[outcome_col, year_col, cluster_col]).copy()

    # Coerce bins to an ordered *string* categorical with 'untreated' first
    cat = _as_ordered_categorical(df[dose_bin_col], untreated_label="untreated")
    df["dose_bin_cat"] = cat

    # Build bin dummies (drop baseline) - pass the Series from dataframe to ensure index is available
    Xb, bin_param_names, name_map = _build_bin_dummies(df["dose_bin_cat"], drop_first=True)

    # Human-readable labels for the positive (non-baseline) bins in the same order as columns
    inv_name_map = {v: k for k, v in name_map.items()}
    pos_labels_hr = [inv_name_map[c] for c in list(Xb.columns)]

    if len(bin_param_names) == 0 or Xb.shape[1] == 0:
        # No positive bins in data
        empty = pd.DataFrame(columns=[
            "bin", "coef", "se", "p", "ci_low", "ci_high", "N", "G", "p_wcb"
        ])
        return BinAttResult(table=empty, model=None, used=df, name_map={})

    # Assemble design matrix
    y = df[outcome_col].astype(float)

    X_list = [Xb]
    ctrl_cols: List[str] = []
    if controls:
        for c in controls:
            if c in df.columns:
                ctrl_cols.append(c)
        if ctrl_cols:
            X_list.append(df[ctrl_cols].astype(float))

    X = pd.concat(X_list, axis=1)

    # Choose estimator
    if use_linearmodels is None:
        use_linearmodels = _HAVE_LM

    model = None
    res = None

    if use_linearmodels and _HAVE_LM:
        # linearmodels PanelOLS with entity & time effects, clustered SEs
        work = pd.concat([y, X, df[[cluster_col, year_col]]], axis=1)
        work = work.dropna()
        work = work.set_index([cluster_col, year_col])

        exog = work[X.columns]
        endog = work[outcome_col]

        model = PanelOLS(endog, exog, entity_effects=True, time_effects=True)
        # Provide clusters aligned with index
        clusters = work.index.get_level_values(0)
        res = model.fit(cov_type="clustered", clusters=clusters)

        coefs = res.params.reindex(Xb.columns)
        ses = res.std_errors.reindex(Xb.columns)
        pvals = res.pvalues.reindex(Xb.columns)
    else:
        # Fallback: statsmodels OLS with explicit C(FE); may be memory-heavy
        work = pd.concat([y, X, df[[cluster_col, year_col]]], axis=1).dropna()
        # Build formula: y ~ dummies + controls + C(unit) + C(year)
        rhs_terms = list(X.columns)
        rhs_terms.append(f"C({cluster_col})")
        rhs_terms.append(f"C({year_col})")
        formula = f"{outcome_col} ~ {' + '.join(rhs_terms)}"
        model = smf.ols(formula=formula, data=work)
        res = model.fit(cov_type="cluster", cov_kwds={"groups": work[cluster_col], "use_correction": True})

        coefs = res.params.reindex(Xb.columns)
        ses = res.bse.reindex(Xb.columns)
        pvals = res.pvalues.reindex(Xb.columns)

    # Confidence intervals (normal approx)
    ci_low = coefs - 1.96 * ses
    ci_high = coefs + 1.96 * ses

    # Basic counts
    G = int(df[cluster_col].nunique())
    N = int(len(df))

    # Compute per-bin MDE at default target power (analytic_mde_from_se defaults to 80% power)
    mde_vals = [analytic_mde_from_se(float(s), int(G)) if pd.notna(s) else np.nan for s in ses.values]

    out = pd.DataFrame({
        "bin": pos_labels_hr,
        "coef": list(coefs.values),
        "se": list(ses.values),
        "p": list(pvals.values),
        "lo": list(ci_low.values),  # renamed to match summary.py expectations
        "hi": list(ci_high.values),  # renamed to match summary.py expectations
        "n_obs": N,  # renamed to match summary.py expectations
        "clusters": G,  # renamed to match summary.py expectations
        "mde": mde_vals,
        "term": list(coefs.index),
    })

    # Indicate detectability at target power: abs(effect) >= MDE
    try:
        out["below_target_power"] = (out["coef"].abs() < out["mde"]).astype(bool)
    except Exception:
        out["below_target_power"] = np.nan

    # Attach optional WCB p-values
    out["p_wcb"] = np.nan
    if use_wcb and _HAVE_WCB:
        try:
            # Use wcb_att_pvalue_r for each bin coefficient separately
            # Build the data with bin dummies
            wcb_data = pd.concat([df[[outcome_col, cluster_col, year_col]], X], axis=1).dropna()
            base_terms = list(Xb.columns)
            if ctrl_cols:
                base_terms += ctrl_cols
            
            # Compute WCB p-value for each bin coefficient
            from ..helpers.utils import choose_wcb_weights_and_B
            
            wt, BB = choose_wcb_weights_and_B(G, wcb_weights, wcb_B)
            wcb_p_dict = {}
            for term_name in Xb.columns:
                try:
                    p_wcb_val = wcb_att_pvalue_r(
                        wcb_data,
                        outcome=outcome_col,
                        regressors=base_terms,
                        fe=[year_col, cluster_col],
                        cluster=cluster_col,
                        param=term_name,
                        B=int(BB),
                        weights=wt,
                        seed=int(getattr(config, "seed", 123) or 123),
                        impose_null=impose_null,
                    )
                    wcb_p_dict[term_name] = p_wcb_val
                except Exception:
                    wcb_p_dict[term_name] = np.nan
            
            # Map WCB p-values using term column (before dropping it)
            out["p_wcb"] = out["term"].map(wcb_p_dict)
        except Exception:  # pragma: no cover - keep robust
            # If WCB fails, we just leave p_wcb as NaN
            pass

    # Add coef_name column for downstream use (e.g., WCB in study.py)
    # Map bin labels to their parameter names using name_map
    out["coef_name"] = out["bin"].map(name_map)
    
    # Final tidy table (drop internal term names, but keep coef_name for WCB)
    out = out.drop(columns=["term"])  # keep only human-readable bin labels

    # Order by the original categorical order of pos_labels (human-readable)
    order_map = {lab: i for i, lab in enumerate(pos_labels_hr)}
    out["_ord"] = out["bin"].map(order_map)
    out = out.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)

    # Return dataclass
    # Build used dataframe with bin dummies for WCB
    used_cols = [outcome_col, dose_bin_col, "dose_bin_cat", year_col, cluster_col]
    if ctrl_cols:
        used_cols += ctrl_cols
    # Start with base columns from df
    used = df[[c for c in used_cols if c in df.columns]].copy()
    # Add bin dummy columns (Xb is aligned with df by index)
    for col in Xb.columns:
        used[col] = Xb[col]

    return BinAttResult(table=out, model=res, used=used, name_map=name_map)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def format_binned_att_table(tbl: pd.DataFrame, digits: int = 3, include_wcb: bool = True) -> pd.DataFrame:
    """Return a copy with rounded numbers and compact CI strings for display."""
    t = tbl.copy()
    for c in ["coef", "se", "p", "ci_low", "ci_high", "p_wcb"]:
        if c in t:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    t["coef"] = t["coef"].round(digits)
    t["se"] = t["se"].round(digits)
    t["p"] = t["p"].round(3)
    if "p_wcb" in t and include_wcb:
        t["p_wcb"] = t["p_wcb"].round(3)
    if "ci_low" in t and "ci_high" in t:
        t["CI (95%)"] = t.apply(lambda r: f"[{r['ci_low']:.{digits}f}, {r['ci_high']:.{digits}f}]" if pd.notna(r['ci_low']) else "", axis=1)
        t = t.drop(columns=["ci_low", "ci_high"], errors="ignore")
    # Reorder
    cols = ["bin", "coef", "se", "p"]
    if "p_wcb" in t and include_wcb:
        cols += ["p_wcb"]
    cols += ["CI (95%)", "N", "G"]
    cols = [c for c in cols if c in t.columns]
    t = t[cols]
    return t


def _ensure_series(x: Iterable | pd.Series | pd.DataFrame) -> pd.Series:
    """Coerce a 1-D input (Series / single-col DataFrame / array / list) to Series."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame for binning.")
        return x.iloc[:, 0]
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)


def _bin_categorical(
    series: pd.Series | pd.DataFrame | Iterable,
    *,
    untreated_label: str = "untreated",
    order: Optional[List[str]] = None,
) -> pd.Categorical:
    """
    Convert a 'dose_bin'-like input into an **ordered** string categorical.
    Accepts Series, single-col DataFrame, numpy array, or list.
    NA -> untreated_label. Works for Interval dtype (stringifies cleanly).
    """
    s = _ensure_series(series)
    obj = s.astype(object).where(s.notna(), other=untreated_label)
    values = obj.astype(str)  # Intervals stringify fine here

    cats_obs = pd.Index(pd.unique(values))

    if order:
        ordered = [c for c in order if c in set(cats_obs)]
        cats = pd.Index(ordered) if ordered else cats_obs
    else:
        cats = cats_obs

    return pd.Categorical(values, categories=cats, ordered=True)


def _build_bin_dummies(
    cat: pd.Categorical,
    *,
    drop_first: bool = True,
    prefix: str = "trbin",
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Build numeric dummies for each category; drop the first as baseline.

    Returns
    -------
    Xb : pd.DataFrame
        DataFrame of dummy regressors with sanitized column names safe for Patsy.
    bin_param_names : List[str]
        The ordered list of parameter names corresponding to positive (non-baseline) bins.
    name_map : Dict[str, str]
        Mapping from the original human-readable bin label -> sanitized column name.
    """
    if not isinstance(cat, pd.Categorical):
        cat = pd.Categorical(_ensure_series(cat).astype(str), ordered=True)

    labels = list(cat.categories)
    if not labels:
        n = len(pd.Series(cat))
        return pd.DataFrame(index=pd.RangeIndex(n)), [], {}

    s = pd.Series(cat).astype(str)
    data: Dict[str, Any] = {}
    all_cols: List[str] = []
    name_map: Dict[str, str] = {}

    for lab in labels:
        lab_str = str(lab)
        safe = f"{prefix}_{_sanitize_bin_label(lab_str)}"
        name_map[lab_str] = safe
        data[safe] = (s == lab_str).astype(float)
        all_cols.append(safe)

    X = pd.DataFrame(data, index=s.index)

    # Drop baseline column if requested (first category by construction)
    bin_param_names: List[str]
    if drop_first and len(labels) > 0:
        ref_label = str(labels[0])
        ref_col = name_map[ref_label]
        X = X.drop(columns=[ref_col])
        bin_param_names = [c for c in all_cols if c != ref_col]
    else:
        bin_param_names = list(all_cols)

    return X, bin_param_names, name_map
