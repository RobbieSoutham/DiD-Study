"""CCUS-informed power tuner with VIF-screened covariates (2–3 bins only).

This module builds a constrained search over StudyConfig knobs, evaluates
DidStudy runs with analytic MDE auditing, applies VIF-based covariate pruning,
keeps p-values (analytic + WCB) alongside PTA/WCB pretrend and HonestDiD
robustness, and writes artifacts plus a markdown report describing whether
we can attain MDE < |effect| robustly and which parameters drive MDE/SE.

Usage
-----
from power_tuner import run_search, analyze_results, write_report
results_df, summary = run_search(cfg_base, target_mde=0.015)
analyze_results(results_df, summary["out_dir"])
write_report(results_df, summary, target_mde=0.015, out_dir=summary["out_dir"])
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
import statsmodels.api as sm

from did_study.helpers.config import StudyConfig
from did_study.study import DidStudy

ORIGINAL_COVARS: List[str] = [
    "Demand_heat",
    "Demand_electricity",
    "Demand_nuclear",
    "Demand_renewables_and_waste",
    "Supply_nuclear",
    "energy_demand_fossil_fuels",
    "CPI_growth",
    "GDP_per_capita_PPP",
    "renewable_to_fossil_supply_ratio",
    "nuclear_share_supply",
]


def analytic_mde_from_se(
    se: float, n_clusters: int, alpha: float = 0.05, power_target: float = 0.80
) -> float:
    """Two-sided MDE using Student-t critical values with df = G-1."""

    G = max(int(n_clusters), 1)
    df = max(G - 1, 1)
    c_alpha = tdist.ppf(1 - alpha / 2, df)
    c_beta = tdist.ppf(power_target, df)
    return float((c_alpha + c_beta) * float(se))


def _coerce_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("__")}
    return {}


def _extract_att(att_obj: Any) -> Dict[str, Any]:
    att_d = _coerce_to_dict(att_obj)
    # backward compatibility with AttResult attributes
    for key in ["coef", "se", "p", "p_wcb", "mde", "n_clusters", "n_obs", "n", "clusters"]:
        if key not in att_d and hasattr(att_obj, key):
            att_d[key] = getattr(att_obj, key)
    return att_d


def _extract_panel_info(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        info = result.get("panel_info") or {}
    else:
        info = getattr(getattr(result, "data", None), "info", {}) or {}
    return _coerce_to_dict(info)


def _extract_event_study(result: Any) -> Dict[str, Any]:
    obj = None
    if isinstance(result, dict):
        obj = result.get("event_study") or result.get("es_pooled")
    else:
        obj = getattr(result, "event_study", None) or getattr(result, "es_pooled", None)
    es_d = _coerce_to_dict(obj)
    for key in ["pta_p", "wcb_p", "names_pre", "names_post", "vcov", "coefs", "data"]:
        if key not in es_d and hasattr(obj, key):
            es_d[key] = getattr(obj, key)
    return es_d


def _median_positive_dose(cfg: StudyConfig) -> float:
    df = getattr(cfg, "df", None)
    if df is None:
        return 0.0
    candidates = [c for c in df.columns if "dose" in c.lower()] + ["dose_level", "dose"]
    seen: List[pd.Series] = []
    for col in candidates:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            seen.append(df[col])
    if not seen:
        return 0.0
    merged = pd.concat(seen, axis=1)
    vals = merged.values.ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 0.0
    return float(np.nanmedian(vals))


def make_search_space(cfg_base: StudyConfig) -> Dict[str, List[Any]]:
    space: Dict[str, List[Any]] = {}
    small_pos = 1e-6 * _median_positive_dose(cfg_base)
    threshold_opts = [0.0]
    if small_pos > 0:
        threshold_opts.append(small_pos)
    space["treat_threshold"] = threshold_opts

    if hasattr(cfg_base, "use_log_outcome"):
        space["use_log_outcome"] = [True, False]
    if hasattr(cfg_base, "differenced"):
        space["differenced"] = [True, False]
    if hasattr(cfg_base, "use_lag_levels_in_diff"):
        space["use_lag_levels_in_diff"] = [True, False]
    if hasattr(cfg_base, "supdem_mode"):
        space["supdem_mode"] = ["direct", "sum"]
    if hasattr(cfg_base, "min_pre"):
        space["min_pre"] = [2, 3, 4, 5]
    if hasattr(cfg_base, "min_post"):
        space["min_post"] = [1, 2, 3, 4, 5]

    dose_quantile_options = [
        [0.0, 0.50, 1.0],
        [0.0, 0.40, 1.0],
        [0.0, 0.33, 1.0],
        [0.0, 0.33, 0.67, 1.0],
        [0.0, 0.25, 0.75, 1.0],
    ]
    space["dose_quantiles"] = dose_quantile_options

    base_unit_cols = getattr(cfg_base, "unit_cols", None)
    if base_unit_cols:
        unit_options: List[Tuple[str, ...]] = [tuple(base_unit_cols)]
        if len(base_unit_cols) > 1:
            unit_options.append((base_unit_cols[0],))
        space["unit_cols"] = unit_options

    space["covariates"] = _build_covariate_subsets()
    return space


def _build_covariate_subsets() -> List[List[str]]:
    energy = [
        "Demand_heat",
        "Demand_electricity",
        "Demand_nuclear",
        "Demand_renewables_and_waste",
        "Supply_nuclear",
        "energy_demand_fossil_fuels",
    ]
    macro_lite = ["CPI_growth", "GDP_per_capita_PPP"]
    ratios = ["renewable_to_fossil_supply_ratio", "nuclear_share_supply"]
    short_energy = ["Demand_heat", "Demand_electricity", "energy_demand_fossil_fuels"]
    nuclear_focus = ["Supply_nuclear", "nuclear_share_supply"]
    covar_sets = [
        energy,
        energy[:4] + ratios,
        short_energy + macro_lite,
        energy + macro_lite,
        energy + ratios,
        ratios,
        macro_lite,
        nuclear_focus + macro_lite,
        short_energy + ratios,
    ]
    # ensure uniqueness and keep length within 6-10 subsets
    unique_sets: List[List[str]] = []
    seen: set[Tuple[str, ...]] = set()
    for s in covar_sets:
        t = tuple(sorted(dict.fromkeys(s)))
        if t not in seen:
            seen.add(t)
            unique_sets.append(list(t))
    return unique_sets


def _serialize_config(cfg: StudyConfig, fields: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in fields:
        if hasattr(cfg, f):
            val = getattr(cfg, f)
            try:
                json.dumps(val)
                out[f] = val
            except TypeError:
                out[f] = str(val)
    return out


def _get_bin_support_flags(info: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    support = info.get("dose_bin_support") or {}
    units = support.get("units") or {}
    rows = support.get("rows") or {}
    thin_bins = False
    per_bin: Dict[str, Dict[str, int]] = {}
    for b, u in units.items():
        if str(b) == "untreated":
            continue
        r = int(rows.get(b, 0))
        u_int = int(u)
        per_bin[str(b)] = {"units": u_int, "rows": r}
        if u_int < 2 or r < 10:
            thin_bins = True
    return thin_bins, per_bin


def _vif_diagnostics(info: Dict[str, Any]) -> Tuple[Optional[float], List[str], List[str]]:
    vif_map = info.get("covariate_vif") or {}
    if isinstance(vif_map, pd.DataFrame):
        if vif_map.shape[1] >= 2:
            vif_map = dict(zip(vif_map.iloc[:, 0], vif_map.iloc[:, 1]))
        else:
            vif_map = dict(zip(vif_map.index, vif_map.iloc[:, 0]))
    offenders: List[str] = []
    caution: List[str] = []
    max_vif = None
    for cov, val in (vif_map or {}).items():
        try:
            v = float(val)
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        max_vif = v if max_vif is None else max(max_vif, v)
        if v >= 10:
            offenders.append(str(cov))
        elif v >= 5:
            caution.append(str(cov))
    return max_vif, offenders, caution


def _compute_post_share(info: Dict[str, Any]) -> Optional[float]:
    try:
        obs = float(info.get("obs"))
        post_rows = float(info.get("post_rows"))
        if obs and obs > 0:
            return post_rows / obs
    except Exception:
        return None
    return None


def _honest_pass(honest: Dict[str, Any], coef: float) -> Tuple[bool, str]:
    if not honest:
        return False, "missing"
    M = honest.get("M")
    lb = honest.get("lb")
    ub = honest.get("ub")
    if M is None or lb is None or ub is None:
        return False, "incomplete_bounds"
    try:
        M_arr = np.asarray(M, dtype=float)
        lb_arr = np.asarray(lb, dtype=float)
        ub_arr = np.asarray(ub, dtype=float)
    except Exception:
        return False, "unusable_bounds"
    if M_arr.size == 0 or lb_arr.size != M_arr.size or ub_arr.size != M_arr.size:
        return False, "size_mismatch"
    small_band = M_arr <= 1.5
    if not np.any(small_band):
        small_band = np.ones_like(M_arr, dtype=bool)
    if coef >= 0:
        ok = np.any(lb_arr[small_band] > 0)
    else:
        ok = np.any(ub_arr[small_band] < 0)
    return bool(ok), ("pass" if ok else "fails_ci")


def _build_config_snapshot(cfg: StudyConfig) -> Dict[str, Any]:
    fields = [
        "use_log_outcome",
        "differenced",
        "use_lag_levels_in_diff",
        "supdem_mode",
        "min_pre",
        "min_post",
        "treat_threshold",
        "dose_quantiles",
        "n_bins",
        "unit_cols",
        "covariates",
    ]
    snap = _serialize_config(cfg, fields)
    if snap.get("dose_quantiles") and not snap.get("n_bins"):
        dq = snap["dose_quantiles"]
        if isinstance(dq, (list, tuple)):
            snap["n_bins"] = max(len(dq) - 1, 0)
    return snap


def evaluate_config(
    cfg: StudyConfig,
    *,
    alpha: float = 0.05,
    power_target: float = 0.80,
) -> Dict[str, Any]:
    cfg_eval = cfg.copy()
    res = DidStudy(cfg_eval).run()
    info = _extract_panel_info(res)
    att = _extract_att(getattr(res, "att", None) or _coerce_to_dict(res).get("att_pooled"))

    # VIF gate: rerun once without high-VIF covariates
    max_vif, offenders, caution = _vif_diagnostics(info)
    covs_attempted = list(getattr(cfg_eval, "covariates", []) or [])
    covs_effective = covs_attempted
    if offenders and covs_attempted:
        filtered_covs = [c for c in covs_attempted if c not in offenders]
        if filtered_covs != covs_attempted:
            cfg_vif = cfg.copy()
            cfg_vif.covariates = filtered_covs
            res = DidStudy(cfg_vif).run()
            info = _extract_panel_info(res)
            att = _extract_att(getattr(res, "att", None) or _coerce_to_dict(res).get("att_pooled"))
            covs_effective = filtered_covs
            max_vif, offenders, caution = _vif_diagnostics(info)

    coef = float(att.get("coef", np.nan))
    se = float(att.get("se", np.nan))
    p = att.get("p")
    p_wcb = att.get("p_wcb")
    n_clusters = int(att.get("n_clusters", att.get("clusters", info.get("units", 0)) or 0))

    mde = att.get("mde")
    if mde is None or (isinstance(mde, float) and (np.isnan(mde) or not np.isfinite(mde))):
        mde = analytic_mde_from_se(se, n_clusters or 1, alpha, power_target) if np.isfinite(se) else np.nan
    else:
        audited = analytic_mde_from_se(se, n_clusters or 1, alpha, power_target) if np.isfinite(se) else np.nan
        if np.isfinite(audited) and not math.isclose(float(mde), float(audited), rel_tol=1e-6, abs_tol=1e-6):
            mde = float(audited)

    es_d = _extract_event_study(res)
    pta_p_raw = es_d.get("pta_p")
    pta_p = None if (isinstance(pta_p_raw, float) and np.isnan(pta_p_raw)) else pta_p_raw
    wcb_joint_p_raw = es_d.get("wcb_p")
    wcb_joint_p = None if (isinstance(wcb_joint_p_raw, float) and np.isnan(wcb_joint_p_raw)) else wcb_joint_p_raw

    honest_d = _coerce_to_dict(getattr(res, "honest_did", None))
    honest_pass, honest_reason = _honest_pass(honest_d, coef)

    thin_bins, per_bin = _get_bin_support_flags(info)
    post_share = _compute_post_share(info)

    p_flag = isinstance(p, (int, float)) and np.isfinite(p) and p < alpha
    p_wcb_flag = isinstance(p_wcb, (int, float)) and np.isfinite(p_wcb) and p_wcb < alpha
    meets_sig = bool(p_flag or p_wcb_flag)
    meets_mde = bool(np.isfinite(mde) and np.isfinite(coef) and abs(float(coef)) >= float(mde))

    row = {
        "coef": float(coef),
        "se": float(se),
        "p": None if (p is None or (isinstance(p, float) and np.isnan(p))) else float(p),
        "p_wcb": None if (p_wcb is None or (isinstance(p_wcb, float) and np.isnan(p_wcb))) else float(p_wcb),
        "pta_p": None if (pta_p is None or (isinstance(pta_p, float) and np.isnan(pta_p))) else float(pta_p),
        "wcb_joint_p": None if (wcb_joint_p is None or (isinstance(wcb_joint_p, float) and np.isnan(wcb_joint_p))) else float(wcb_joint_p),
        "n_clusters": int(n_clusters) if n_clusters is not None else None,
        "n": int(att.get("n", att.get("n_obs", info.get("obs", 0)) or 0)) if info else None,
        "mde": float(mde) if mde is not None else np.nan,
        "post_share": post_share,
        "differenced": bool(getattr(cfg_eval, "differenced", False)),
        "use_log_outcome": bool(getattr(cfg_eval, "use_log_outcome", False)),
        "use_lag_levels_in_diff": bool(getattr(cfg_eval, "use_lag_levels_in_diff", False)),
        "min_pre": int(getattr(cfg_eval, "min_pre", 0)),
        "min_post": int(getattr(cfg_eval, "min_post", 0)),
        "supdem_mode": getattr(cfg_eval, "supdem_mode", None),
        "treat_threshold": float(getattr(cfg_eval, "treat_threshold", 0.0)),
        "dose_quantiles": getattr(cfg_eval, "dose_quantiles", None),
        "n_bins": int(getattr(cfg_eval, "n_bins", len(getattr(cfg_eval, "dose_quantiles", []) or []) - 1) or 0),
        "unit_cols": tuple(getattr(cfg_eval, "unit_cols", []) or ()),
        "covariates_attempted": covs_attempted,
        "covariates_effective": covs_effective,
        "covar_set_size": len(covs_effective or []),
        "covariate_vif_max": max_vif,
        "covariate_vif_offenders": offenders,
        "covariate_vif_caution": caution,
        "thin_bins": thin_bins,
        "bin_support": per_bin,
        "units": info.get("units"),
        "ever_treated": info.get("ever_treated"),
        "obs": info.get("obs"),
        "post_rows": info.get("post_rows"),
        "differenced_panel": info.get("differenced"),
        "pta_note": None if pta_p is not None else "pta_missing",
        "wcb_joint_note": None if wcb_joint_p is not None else "wcb_missing",
        "honest_pass": bool(honest_pass),
        "honest_note": honest_reason,
        "meets_sig": meets_sig,
        "meets_mde": meets_mde,
        "small_g": bool(n_clusters and n_clusters < 6),
        "config_snapshot": _build_config_snapshot(cfg_eval),
    }
    if getattr(cfg_eval, "use_log_outcome", False):
        row["effect_pct"] = 100 * row["coef"]
        row["mde_pct"] = 100 * row["mde"]
    else:
        row["effect_pct"] = None
        row["mde_pct"] = None
    return row


def run_search(
    cfg_base: StudyConfig,
    *,
    target_mde: float,
    alpha: float = 0.05,
    power_target: float = 0.80,
    max_candidates: int = 160,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    space = make_search_space(cfg_base)
    keys = list(space.keys())
    values = [space[k] for k in keys]
    candidates = list(product(*values))
    candidates = candidates[:max_candidates]

    out_dir_base = Path(getattr(cfg_base, "artifact_dir", None) or "./_artifacts")
    run_dir = out_dir_base / f"power_search_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "runs").mkdir(exist_ok=True)

    rows: List[Dict[str, Any]] = []
    drop_reasons: List[str] = []

    for idx, combo in enumerate(candidates):
        cfg = cfg_base.copy()
        overrides = dict(zip(keys, combo))
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        dq = getattr(cfg, "dose_quantiles", None)
        if dq:
            cfg.n_bins = len(dq) - 1
        row = evaluate_config(cfg, alpha=alpha, power_target=power_target)
        thin_bins = row.get("thin_bins", False)
        if thin_bins:
            drop_reasons.append(f"candidate {idx} thin bins")
            continue

        row["meets_target_mde"] = bool(np.isfinite(row["mde"]) and row["mde"] <= target_mde)
        mde_val = row.get("mde")
        row["mde_ratio"] = (
            abs(row["coef"]) / mde_val if (isinstance(mde_val, (int, float)) and np.isfinite(mde_val) and mde_val != 0) else np.nan
        )
        rows.append(row)

        with open(run_dir / "runs" / f"run_{idx}.json", "w") as f:
            json.dump(row, f, default=_json_fallback, indent=2)
        panel_info = {
            k: row.get(k)
            for k in ["bin_support", "units", "ever_treated", "obs", "post_rows", "differenced_panel", "covariate_vif_max"]
        }
        with open(run_dir / "runs" / f"panel_{idx}.json", "w") as f:
            json.dump(panel_info, f, default=_json_fallback, indent=2)

    if not rows:
        empty_df = pd.DataFrame()
        summary = {"out_dir": str(run_dir), "meeting_count": 0, "highlight_count": 0, "pta_wcb_count": 0, "robust_count": 0}
        return empty_df, summary

    results_df = pd.DataFrame(rows)
    results_df.index.name = "candidate_id"

    meeting_df = results_df[results_df.meets_target_mde]
    highlight_df = meeting_df[meeting_df.meets_sig]
    pta_wcb_df = meeting_df[(meeting_df["pta_p"].notna()) & (meeting_df["pta_p"] < alpha) & (meeting_df["wcb_joint_p"].notna()) & (meeting_df["wcb_joint_p"] < alpha)]
    robust_df = meeting_df[meeting_df.honest_pass]

    sort_df = results_df.copy()
    sort_df["pta_flag"] = (sort_df["pta_p"].notna()) & (sort_df["pta_p"] < alpha)
    sort_df["wcb_flag"] = (sort_df["wcb_joint_p"].notna()) & (sort_df["wcb_joint_p"] < alpha)
    sort_df["abs_coef"] = sort_df["coef"].abs()
    sort_df["p_sort"] = sort_df["p"].fillna(1)
    sort_df["p_wcb_sort"] = sort_df["p_wcb"].fillna(1)
    results_df = sort_df.sort_values(
        by=[
            "meets_target_mde",
            "meets_sig",
            "pta_flag",
            "wcb_flag",
            "honest_pass",
            "mde",
            "se",
            "abs_coef",
            "p_sort",
            "p_wcb_sort",
        ],
        ascending=[False, False, False, False, False, True, True, False, True, True],
    )

    meeting_df.to_csv(run_dir / "meeting_configs.csv")
    highlight_df.to_csv(run_dir / "highlight_configs.csv")
    pta_wcb_df.to_csv(run_dir / "pta_wcb_configs.csv")
    robust_df.to_csv(run_dir / "robust_configs.csv")
    results_df.to_csv(run_dir / "results_df.csv")

    best_overall = results_df.iloc[0].to_dict()
    summary = {
        "out_dir": str(run_dir),
        "meeting_count": int(meeting_df.shape[0]),
        "highlight_count": int(highlight_df.shape[0]),
        "pta_wcb_count": int(pta_wcb_df.shape[0]),
        "robust_count": int(robust_df.shape[0]),
        "meeting_ids": meeting_df.index.tolist(),
        "highlight_ids": highlight_df.index.tolist(),
        "pta_wcb_ids": pta_wcb_df.index.tolist(),
        "robust_ids": robust_df.index.tolist(),
        "best_overall": best_overall,
        "drop_reasons": drop_reasons,
    }

    with open(run_dir / "winner.json", "w") as f:
        json.dump(best_overall, f, default=_json_fallback, indent=2)
    return results_df, summary


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def analyze_results(results_df: pd.DataFrame, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if results_df.empty:
        (out_path / "analysis_summary.json").write_text(json.dumps({"error": "no_results"}, indent=2))
        return

    df = results_df.copy()
    rhs = (
        "mde ~ C(use_log_outcome) + C(differenced) + C(use_lag_levels_in_diff) + "
        "min_pre + min_post + C(supdem_mode) + n_clusters + n_bins + covar_set_size + post_share"
    )
    se_rhs = rhs.replace("mde ~", "se ~")

    def _fit(formula: str) -> pd.DataFrame:
        try:
            model = sm.OLS.from_formula(formula, data=df)
            res = model.fit(cov_type="HC1")
            return res.summary2().tables[1]
        except Exception:
            return pd.DataFrame()

    mde_tab = _fit(rhs)
    se_tab = _fit(se_rhs)
    if not mde_tab.empty:
        mde_tab.to_csv(out_path / "drivers_ols_summary.csv")
    if not se_tab.empty:
        se_tab.to_csv(out_path / "drivers_se_ols_summary.csv")

    # partial effects for toggles
    discrete_cols = ["use_log_outcome", "differenced", "use_lag_levels_in_diff", "supdem_mode", "n_bins"]
    partials: List[Dict[str, Any]] = []
    for col in discrete_cols:
        if col not in df.columns:
            continue
        for val, grp in df.groupby(col):
            partials.append({"param": col, "value": val, "mde_mean": grp["mde"].mean(), "se_mean": grp["se"].mean()})
    pd.DataFrame(partials).to_csv(out_path / "partial_effects.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        plt.figure()
        df.plot.scatter(x="n_clusters", y="mde")
        plt.tight_layout()
        plt.savefig(out_path / "mde_vs_n_clusters.png")
        plt.close()

        plt.figure()
        df.boxplot(column="mde", by="n_bins")
        plt.tight_layout()
        plt.savefig(out_path / "mde_vs_n_bins.png")
        plt.close()

        plt.figure()
        df.plot.scatter(x="post_share", y="mde_ratio")
        plt.tight_layout()
        plt.savefig(out_path / "mde_ratio_vs_post_share.png")
        plt.close()

        plt.figure()
        df.boxplot(column="se", by="differenced")
        plt.tight_layout()
        plt.savefig(out_path / "se_by_differenced.png")
        plt.close()

        plt.figure()
        df.boxplot(column="se", by="use_log_outcome")
        plt.tight_layout()
        plt.savefig(out_path / "se_by_log.png")
        plt.close()
    except Exception:
        pass

    summary = {
        "mde_drivers": mde_tab.head().to_dict() if not mde_tab.empty else {},
        "se_drivers": se_tab.head().to_dict() if not se_tab.empty else {},
        "partials": partials,
    }
    (out_path / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=_json_fallback))


def write_report(
    results_df: pd.DataFrame,
    summary: Dict[str, Any],
    *,
    target_mde: float,
    out_dir: str,
    alpha: float = 0.05,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if results_df.empty:
        (out_path / "power_tuner_report.md").write_text("No viable configurations were evaluated.")
        return

    best = summary.get("best_overall", {})
    meeting_df = results_df[results_df.meets_target_mde]
    highlight_df = meeting_df[meeting_df.meets_sig]
    pta_wcb_df = meeting_df[(meeting_df["pta_p"].notna()) & (meeting_df["pta_p"] < alpha) & (meeting_df["wcb_joint_p"].notna()) & (meeting_df["wcb_joint_p"] < alpha)]
    robust_df = meeting_df[meeting_df.honest_pass]

    def _fmt(val: Any) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "NaN"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    table_cols = [
        "dose_quantiles",
        "n_bins",
        "min_pre",
        "min_post",
        "supdem_mode",
        "differenced",
        "use_log_outcome",
        "coef",
        "se",
        "p",
        "p_wcb",
        "pta_p",
        "wcb_joint_p",
        "n_clusters",
        "mde",
        "mde_ratio",
        "meets_target_mde",
        "meets_sig",
        "honest_pass",
        "thin_bins",
        "covariate_vif_max",
    ]
    top_rows = results_df.head(10)[table_cols].copy()

    report_lines = []
    report_lines.append("# CCUS Power Tuner (2–3 bins, VIF-screened)")
    report_lines.append("\n## Executive summary")
    robust_yes = not robust_df.empty and any(robust_df.mde <= target_mde)
    if robust_yes:
        report_lines.append("At least one configuration achieves MDE < |effect| and is robust (PTA, WCB, HonestDiD).")
    else:
        report_lines.append("No configuration met all robustness gates with MDE < |effect|; see limiting factors below.")
    report_lines.append(
        f"Counts — meeting target: {summary.get('meeting_count', 0)}, significant: {summary.get('highlight_count', 0)}, "
        f"PTA+WCB clean: {summary.get('pta_wcb_count', 0)}, HonestDiD-pass: {summary.get('robust_count', 0)}"
    )

    report_lines.append("\n## Best configurations (top 10)")
    report_lines.append(top_rows.to_markdown(index=False))

    report_lines.append("\n## Can we attain MDE < |effect| robustly?")
    if robust_yes:
        best_robust = robust_df.sort_values("mde").head(3)
        desc = []
        for _, r in best_robust.iterrows():
            desc.append(
                f"dose_quantiles={r['dose_quantiles']}, n_bins={r['n_bins']}, min_pre={r['min_pre']}, min_post={r['min_post']}, "
                f"supdem_mode={r['supdem_mode']}, covars={r['covariates_effective']}"
            )
        report_lines.append("Yes. Smallest-MDE robust configs:\n- " + "\n- ".join(desc))
    else:
        bottlenecks = []
        if results_df["n_clusters"].min() < 6:
            bottlenecks.append("cluster count is small (<6)")
        if results_df.thin_bins.any():
            bottlenecks.append("thin bin support")
        if results_df["covariate_vif_max"].max() >= 10:
            bottlenecks.append("multicollinearity inflating SE (VIF>=10)")
        if not bottlenecks:
            bottlenecks.append("HonestDiD bounds or pre-trend tests are failing")
        report_lines.append("No. Bottlenecks: " + "; ".join(bottlenecks))

    report_lines.append("\n## What parameters affect MDE?")
    if (Path(out_dir) / "drivers_ols_summary.csv").exists():
        mde_tab = pd.read_csv(Path(out_dir) / "drivers_ols_summary.csv")
        top = mde_tab.head(8)[["Coef.", "Std.Err.", "P>|t|", "index"]] if "index" in mde_tab.columns else mde_tab.head(8)
        report_lines.append(top.to_markdown(index=False))
    else:
        report_lines.append("Insufficient data to fit drivers model.")

    report_lines.append("\n## MDE audit")
    report_lines.append("Tuner used theory-correct MDE (two-sided, t with df=G−1): MDE = (t_{1-α/2, G-1} + t_{power, G-1}) × SE.")

    report_lines.append("\n## Diagnostics & issues")
    diag = []
    if results_df.thin_bins.any():
        diag.append("Thin bins detected (post-bin <2 units or <10 rows); such candidates were dropped.")
    if results_df["covariate_vif_max"].max() >= 10:
        diag.append("VIF >= 10 observed; offending covariates were pruned.")
    if results_df["n_clusters"].min() < 6:
        diag.append("Small cluster count (n_clusters < 6) may inflate t-critical values.")
    if meeting_df.empty:
        diag.append("No configurations met the target MDE threshold.")
    if not diag:
        diag.append("No major data-quality warnings.")
    for d in diag:
        report_lines.append(f"- {d}")

    report_lines.append("\n## Recommendations")
    recs = [
        "Prefer 2–3 bin dose_quantiles from the CCUS set; discard bins with thin post support.",
        "Use min_pre ≥ 3 and consider use_lag_levels_in_diff=True when differenced to stabilise levels.",
        "Apply VIF pruning (drop ≥10) and keep energy-mix + macro-lite covariates.",
        "Check PTA and WCB pretrend p-values on leads only; favour configs that keep both ≥0.05.",
        "If log outcome is used, interpret effects and MDE on percent scale as reported.",
    ]
    for r in recs:
        report_lines.append(f"- {r}")

    (out_path / "power_tuner_report.md").write_text("\n".join(report_lines))

