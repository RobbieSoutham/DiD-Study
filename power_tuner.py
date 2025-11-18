from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import t as tdist  # type: ignore
except Exception:  # pragma: no cover
    tdist = None

from did_study.helpers.config import StudyConfig
from did_study.study import DidStudy
import hashlib
from pathlib import Path

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

def analytic_mde_from_se(se: float, n_clusters: int, alpha: float = 0.05, power_target: float = 0.80) -> float:
    G = max(int(n_clusters), 1)
    df = max(G - 1, 1)
    if tdist is not None:
        c_alpha = tdist.ppf(1 - alpha / 2, df)
        c_beta = tdist.ppf(power_target, df)
        crit = float(c_alpha + c_beta)
    else:
        from statistics import NormalDist  # type: ignore
        z = NormalDist().inv_cdf
        crit = float(z(1 - alpha / 2) + z(power_target))
    return float(abs(se) * crit)

def make_search_space(cfg_base: StudyConfig) -> Dict[str, List[Any]]:
    grid: Dict[str, List[Any]] = {}
    if hasattr(cfg_base, "use_log_outcome"):
        grid["use_log_outcome"] = [True, False]
    if hasattr(cfg_base, "differenced"):
        grid["differenced"] = [True, False]
    if hasattr(cfg_base, "use_lag_levels_in_diff"):
        grid["use_lag_levels_in_diff"] = [True, False]
    if hasattr(cfg_base, "supdem_mode"):
        grid["supdem_mode"] = ["direct", "sum"]
    if hasattr(cfg_base, "outcome_mode"):
        grid["outcome_mode"] = [getattr(cfg_base, "outcome_mode", "direct"), "total", "direct"]
    if hasattr(cfg_base, "min_pre"):
        grid["min_pre"] = [2, 3, 4, 5]
    if hasattr(cfg_base, "min_post"):
        grid["min_post"] = [1, 2, 3, 4, 5]
    if hasattr(cfg_base, "treat_threshold"):
        grid["treat_threshold"] = [0, "small_pos"]
    if hasattr(cfg_base, "dose_quantiles"):
        grid["dose_quantiles"] = [
            [0.0, 0.50, 1.0],
            [0.0, 0.40, 1.0],
            [0.0, 0.33, 1.0],
            [0.0, 0.33, 0.67, 1.0],
            [0.0, 0.25, 0.75, 1.0],
        ]
    base_sets: List[List[str]] = []
    base_sets.append([
        "renewable_to_fossil_supply_ratio",
        "energy_demand_fossil_fuels",
        "nuclear_share_supply",
    ])
    base_sets.append([
        "renewable_to_fossil_supply_ratio",
        "energy_demand_fossil_fuels",
        "nuclear_share_supply",
        "GDP_per_capita_PPP",
        "CPI_growth",
    ])
    base_sets.append([
        "GDP_per_capita_PPP",
        "CPI_growth",
        "energy_demand_fossil_fuels",
        "nuclear_share_supply",
    ])
    base_sets.append([
        "renewable_to_fossil_supply_ratio",
        "nuclear_share_supply",
    ])
    base_sets.append(["nuclear_share_supply"])
    base_sets.append([
        "Demand_heat",
        "Demand_electricity",
        "Demand_renewables_and_waste",
        "energy_demand_fossil_fuels",
        "renewable_to_fossil_supply_ratio",
        "nuclear_share_supply",
    ])
    base_sets.append([
        "Demand_electricity",
        "energy_demand_fossil_fuels",
        "renewable_to_fossil_supply_ratio",
        "nuclear_share_supply",
        "GDP_per_capita_PPP",
    ])
    base_sets.append([
        "GDP_per_capita_PPP",
        "CPI_growth",
    ])
    base_sets = [[c for c in s if c in ORIGINAL_COVARS] for s in base_sets]
    if hasattr(cfg_base, "covariates"):
        grid["covariates"] = base_sets
    return grid

# Stubs to be filled in below via incremental patches to avoid long filename errors
def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _att_to_dict(att: Any) -> Dict[str, Any]:
    if att is None:
        return {}
    if isinstance(att, dict):
        return dict(att)
    out: Dict[str, Any] = {}
    for k in ("coef", "se", "p", "p_wcb", "mde", "n_clusters", "clusters", "n", "n_obs"):
        if hasattr(att, k):
            out[k] = getattr(att, k)
    return out

def _result_to_panel_info(run_result: Any) -> Dict[str, Any]:
    if hasattr(run_result, "data") and hasattr(run_result.data, "info"):
        return dict(getattr(run_result.data, "info", {}) or {})
    if isinstance(run_result, dict):
        return dict(run_result.get("panel_info", {}) or {})
    return {}

def _panel_stats_from_result(run_result: Any) -> Dict[str, Any]:
    try:
        df = getattr(getattr(run_result, "data", None), "panel", None)
        if df is None or not hasattr(df, "shape"):
            return {}
        obs = int(df.shape[0])
        units = int(df["unit_id"].nunique()) if "unit_id" in df.columns else None
        ever_treated = int(df.groupby("unit_id")["treated_ever"].max().sum()) if "treated_ever" in df.columns else None
        post_rows = int((df.get("post", 0) == 1).sum()) if "post" in df.columns else None
        return {"obs": obs, "units": units, "ever_treated": ever_treated, "post_rows": post_rows}
    except Exception:
        return {}

def parse_vif_from_panel_info(info: Dict[str, Any]) -> Tuple[float | None, List[str], List[str]]:
    cov_vif = info.get("covariate_vif")
    if cov_vif is None:
        return None, [], []
    pairs: List[Tuple[str, float]] = []
    try:
        if isinstance(cov_vif, dict):
            for k, v in cov_vif.items():
                pairs.append((str(k), _as_float(v)))
        elif hasattr(cov_vif, "to_dict"):
            d = cov_vif.to_dict()
            if isinstance(d, dict):
                for k, v in d.items():
                    pairs.append((str(k), _as_float(v)))
    except Exception:
        pairs = []
    if not pairs:
        return None, [], []
    vals = [v for _, v in pairs if np.isfinite(v)]
    vif_max = float(max(vals)) if vals else None
    offenders_ge10 = [k for k, v in pairs if v >= 10]
    offenders_5_10 = [k for k, v in pairs if 5 <= v < 10]
    return vif_max, offenders_ge10, offenders_5_10

def _bin_support_flags(info: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    sup = info.get("dose_bin_support") or {}
    units_map = sup.get("units") or {}
    rows_map = sup.get("rows") or {}
    thin = False
    detail: Dict[str, Dict[str, int]] = {"units": {}, "rows": {}}
    try:
        for b, u in units_map.items():
            u_int = int(u)
            r_int = int(rows_map.get(b, 0))
            detail["units"][str(b)] = u_int
            detail["rows"][str(b)] = r_int
            if str(b) != "untreated":
                if u_int < 2 or r_int < 10:
                    thin = True
    except Exception:
        thin = False
    return thin, detail

def _median_positive_dose(info: Dict[str, Any]) -> float:
    s = info.get("dose_series")
    try:
        if s is not None:
            arr = np.asarray(s, dtype=float)
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if arr.size > 0:
                return float(np.nanmedian(arr))
    except Exception:
        pass
    return float("nan")

def _config_snapshot(cfg: StudyConfig, overrides: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = [
        "use_log_outcome",
        "differenced",
        "use_lag_levels_in_diff",
        "outcome_mode",
        "supdem_mode",
        "min_pre",
        "min_post",
        "treat_threshold",
        "dose_quantiles",
        "n_bins",
        "covariates",
    ]
    for k in keys:
        if hasattr(cfg, k):
            out[k] = getattr(cfg, k)
    out.update({k: v for k, v in overrides.items() if k in keys})
    return out

def _select_panel_info_fields(info: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "differenced",
        "use_lag_levels_in_diff",
        "dose_bin_edges",
        "dose_bin_support",
        "dropped_units",
        "covariates_used",
        "covariate_vif",
        "n_bins",
    ]
    return {k: info.get(k) for k in keys if k in info}

def _json_default(o: Any) -> Any:  # pragma: no cover
    try:
        import numpy as _np
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    try:
        from dataclasses import asdict as _asdict
        return _asdict(o)
    except Exception:
        pass
    try:
        return float(o)
    except Exception:
        pass
    try:
        return str(o)
    except Exception:
        pass
    return None

def evaluate_config(cfg: StudyConfig, *, alpha: float = 0.05, power_target: float = 0.80) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    # First run
    res = DidStudy(cfg).run()
    info = _result_to_panel_info(res)
    att = _att_to_dict(getattr(res, "att", None) or (res.get("att_pooled") if isinstance(res, dict) else None))

    coef = _as_float(att.get("coef"))
    se = _as_float(att.get("se"))
    p = att.get("p") if att.get("p") is not None else np.nan
    p_wcb = att.get("p_wcb") if att.get("p_wcb") is not None else np.nan
    n_clusters = int(att.get("n_clusters") or att.get("clusters") or 0)
    n = int(att.get("n") or att.get("n_obs") or 0)

    # MDE audit / replacement
    mde_att = att.get("mde")
    if mde_att is None or not np.isfinite(mde_att):
        mde_val = analytic_mde_from_se(se, n_clusters, alpha=alpha, power_target=power_target)
        mde_note = "computed_by_tuner"
    else:
        theo = analytic_mde_from_se(se, n_clusters, alpha=alpha, power_target=power_target)
        mde_val = float(mde_att)
        mde_note = "matches_theory" if (np.isfinite(mde_val) and np.isfinite(theo) and abs(mde_val - theo) < 1e-9) else "differs_from_theory"

    # Event-study pretrend (PTA F on pre-leads already computed by estimator)
    es = getattr(res, "event_study", None) or (res.get("event_study") if isinstance(res, dict) else None)
    pta_p = float(getattr(es, "pta_p", np.nan)) if es is not None else np.nan

    # WCB joint on pre-leads only (try only if WCB requested; else NaN)
    wcb_joint_pre = np.nan
    if bool(getattr(cfg, "use_wcb", False)):
        try:
            names_pre: List[str] = list(getattr(es, "names_pre", []) or [])
            used_df = getattr(es, "data", None)
            if names_pre and used_df is not None and not used_df.empty:
                from did_study.robustness.wcb import WildClusterBootstrap, FitSpec, TestSpec

                outcome = getattr(getattr(res, "data", None), "outcome_name", None)
                year_col = getattr(cfg, "year_col", "Year")
                include_unit_fe = not (isinstance(outcome, str) and outcome.lower().startswith("d_"))
                fe_terms = [f"C({year_col})"]
                if include_unit_fe:
                    fe_terms.insert(0, "C(unit_id)")

                covs = list(info.get("covariates_used", []) or [])
                regressors = names_pre + covs

                cluster_spec = getattr(cfg, "cluster_col", None)
                if isinstance(cluster_spec, str):
                    cluster_terms = [cluster_spec]
                else:
                    cluster_terms = list(cluster_spec or [])
                if not cluster_terms:
                    cluster_terms = ["unit_id"]

                keep_cols = [c for c in set([outcome, "unit_id", year_col] + regressors + cluster_terms) if c in used_df.columns]
                runner = WildClusterBootstrap(
                    df=used_df[keep_cols].copy(),
                    fit_spec=FitSpec(outcome=outcome, regressors=regressors, fe=fe_terms, cluster=cluster_terms),
                    B=1999,
                    weights="rademacher",
                    seed=getattr(cfg, "seed", None),
                    impose_null=True,
                )
                wcb_joint_pre = float(runner.pvalue(TestSpec(joint_zero=names_pre)))
        except Exception:
            wcb_joint_pre = np.nan

    # HonestDiD robustness
    honest_pass = False
    honest_note = ""
    hid = getattr(res, "honest_did", None) or (res.get("honestdid") if isinstance(res, dict) else None)
    try:
        if isinstance(hid, dict):
            M = np.asarray(hid.get("M", []), dtype=float)
            lo = np.asarray(hid.get("lo", hid.get("lb", [])), dtype=float)
            hi = np.asarray(hid.get("hi", hid.get("ub", [])), dtype=float)
            if M.size and lo.size and hi.size and M.size == lo.size == hi.size:
                small = M <= 1.0
                if np.any(small):
                    if coef >= 0:
                        honest_pass = bool(np.any(lo[small] > 0))
                    else:
                        honest_pass = bool(np.any(hi[small] < 0))
                else:
                    honest_note = "no small-M grid points"
            else:
                honest_note = "missing or mismatched HonestDiD arrays"
        else:
            honest_note = "HonestDiD not available"
    except Exception:
        honest_pass = False
        honest_note = "HonestDiD parse error"

    panel_stats = _panel_stats_from_result(res)
    obs = panel_stats.get("obs")
    post_rows = panel_stats.get("post_rows")
    post_share = (float(post_rows) / float(obs)) if (post_rows and obs) else np.nan

    thin_bins, bin_details = _bin_support_flags(info)

    # VIF gate: drop VIF>=10 and re-estimate once
    vif_max, vif_ge10, vif_5_10 = parse_vif_from_panel_info(info)
    cov_attempted = list(getattr(cfg, "covariates", []) or [])
    cov_effective = list(cov_attempted)
    if vif_ge10:
        offenders_base = []
        for name in vif_ge10:
            if isinstance(name, str) and (name.startswith("L1_") or name.startswith("d_")):
                offenders_base.append(name.split("_", 1)[1])
            else:
                offenders_base.append(str(name))
        offenders_base = sorted(set(offenders_base))
        cov_effective = [c for c in cov_effective if c not in offenders_base]
        if set(cov_effective) != set(cov_attempted):
            cfg2 = cfg.copy()
            cfg2.covariates = list(cov_effective)
            res2 = DidStudy(cfg2).run()
            res, cfg = res2, cfg2
            info = _result_to_panel_info(res2)
            att = _att_to_dict(getattr(res2, "att", None) or (res2.get("att_pooled") if isinstance(res2, dict) else None))
            coef = _as_float(att.get("coef"))
            se = _as_float(att.get("se"))
            p = att.get("p") if att.get("p") is not None else np.nan
            p_wcb = att.get("p_wcb") if att.get("p_wcb") is not None else np.nan
            n_clusters = int(att.get("n_clusters") or att.get("clusters") or 0)
            n = int(att.get("n") or att.get("n_obs") or 0)
            mde_val = analytic_mde_from_se(se, n_clusters, alpha=alpha, power_target=power_target)
            vif_max, vif_ge10, vif_5_10 = parse_vif_from_panel_info(info)

    meets_sig = (isinstance(p, (int, float)) and p < alpha) or (isinstance(p_wcb, (int, float)) and p_wcb < alpha)
    meets_mde = (np.isfinite(mde_val) and np.isfinite(coef) and abs(coef) >= float(mde_val))
    small_g = (n_clusters < 6) if n_clusters is not None else False

    snap = _config_snapshot(cfg, overrides)
    n_bins = int(snap.get("n_bins") or (len(snap.get("dose_quantiles") or []) - 1) or 0)
    effect_pct = (100.0 * coef) if bool(getattr(cfg, "use_log_outcome", False)) else np.nan
    mde_pct = (100.0 * mde_val) if bool(getattr(cfg, "use_log_outcome", False)) else np.nan

    row: Dict[str, Any] = {
        "coef": coef,
        "se": se,
        "p": p,
        "p_wcb": p_wcb,
        "n": int(n),
        "n_clusters": int(n_clusters),
        "mde": mde_val,
        "mde_audit": mde_note,
        "effect_pct": effect_pct,
        "mde_pct": mde_pct,
        "pta_p": float(pta_p) if pta_p is not None else np.nan,
        "wcb_joint_p": float(wcb_joint_pre) if wcb_joint_pre is not None else np.nan,
        "honest_pass": bool(honest_pass),
        "honest_note": honest_note,
        "obs": panel_stats.get("obs"),
        "post_rows": panel_stats.get("post_rows"),
        "post_share": post_share,
        "thin_bins": bool(thin_bins),
        "small_g": bool(small_g),
        "covariate_vif_max": vif_max if vif_max is not None else np.nan,
        "covariate_vif_ge10": ",".join(map(str, vif_ge10)) if vif_ge10 else "",
        "covariate_vif_5_10": ",".join(map(str, vif_5_10)) if vif_5_10 else "",
        "n_bins": n_bins,
        "config_snapshot": snap,
        "covariates_attempted": ",".join(cov_attempted) if cov_attempted else "",
        "covariates_effective": ",".join(cov_effective) if cov_effective else "",
        "bin_units_min": int(min((_bin_support_flags(info)[1].get("units") or {"_": np.inf}).values())) if _bin_support_flags(info)[1].get("units") else np.nan,
        "bin_rows_min": int(min((_bin_support_flags(info)[1].get("rows") or {"_": np.inf}).values())) if _bin_support_flags(info)[1].get("rows") else np.nan,
        # keep a small panel_info subset for per-run JSON convenience
        "panel_info": _select_panel_info_fields(info),
        "meets_sig": bool(meets_sig),
        "meets_mde": bool(meets_mde),
    }

    # Dataset cache: save prepared panel for this dataset+prep spec
    try:
        base_dir = getattr(cfg, "artifact_dir", None) or "./_artifacts"
        cache_dir = os.path.join(base_dir, "datasets_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        data_path = getattr(cfg, "_data_path", None)
        key_parts = [
            str(data_path or "<mem>"),
            str(getattr(cfg, "outcome_mode", "direct")),
            str(getattr(cfg, "use_log_outcome", True)),
            str(getattr(cfg, "differenced", True)),
            str(getattr(cfg, "use_lag_levels_in_diff", True)),
            str(getattr(cfg, "min_pre", 2)),
            str(getattr(cfg, "min_post", 1)),
            str(getattr(cfg, "supdem_mode", "sum")),
            str(getattr(cfg, "treat_threshold", 0.0)),
            str(getattr(cfg, "dose_quantiles", None)),
            str(getattr(cfg, "dose_bins", None)),
            str(getattr(cfg, "n_bins", None)),
        ]
        cache_key = hashlib.md5("|".join(key_parts).encode("utf-8")).hexdigest()[:16]
        panel_path = os.path.join(cache_dir, f"panel_{cache_key}.parquet")
        info_path = os.path.join(cache_dir, f"panel_{cache_key}.info.json")
        if not os.path.exists(panel_path):
            panel_df = getattr(getattr(res, "data", None), "panel", None)
            if panel_df is not None and hasattr(panel_df, "to_parquet"):
                try:
                    panel_df.to_parquet(panel_path, index=False)
                    with open(info_path, "w", encoding="utf-8") as fh:
                        json.dump(info, fh, indent=2, default=_json_default)
                except Exception:
                    pass
        row["dataset_cache_key"] = cache_key
        row["dataset_cache_path"] = panel_path
    except Exception:
        pass

    return row

def run_search(
    cfg_base: StudyConfig,
    target_mde: float,
    *,
    alpha: float = 0.05,
    power_target: float = 0.80,
    max_candidates: int = 160,
):
    base = cfg_base.copy()
    grid = make_search_space(base)

    root = getattr(base, "artifact_dir", None) or "./_artifacts"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, f"power_search_{ts}")
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)

    # Build candidate cartesian product
    keys: List[str] = [
        k for k in [
            "use_log_outcome",
            "differenced",
            "use_lag_levels_in_diff",
            "outcome_mode",
            "supdem_mode",
            "min_pre",
            "min_post",
            "treat_threshold",
            "dose_quantiles",
            "covariates",
        ] if k in grid
    ]
    values_list: List[List[Any]] = [grid[k] for k in keys]

    # Precompute small_pos
    try:
        base_info = _result_to_panel_info(DidStudy(base).run())
        med_pos = _median_positive_dose(base_info)
        small_pos_val = float(1e-6 * med_pos) if np.isfinite(med_pos) and med_pos > 0 else 0.0
    except Exception:
        small_pos_val = 0.0

    rows: List[Dict[str, Any]] = []
    kept: int = 0
    cand_id: int = 0

    from itertools import product as _product
    for combo in _product(*values_list):
        if cand_id >= max_candidates:
            break
        overrides = dict(zip(keys, combo))
        dq = overrides.get("dose_quantiles") or []
        n_bins = len(dq) - 1 if dq else 0
        if n_bins not in (2, 3):
            cand_id += 1
            continue

        cfg = base.copy()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                continue
            if k == "treat_threshold" and v == "small_pos":
                setattr(cfg, k, small_pos_val)
            else:
                setattr(cfg, k, v)
        cfg.n_bins = n_bins

        try:
            row = evaluate_config(cfg, alpha=alpha, power_target=power_target)
        except Exception as e:  # noqa: BLE001
            row = {"error": str(e), "config_snapshot": _config_snapshot(cfg, overrides)}

        # Filter thin bins
        if bool(row.get("thin_bins", False)):
            try:
                with open(os.path.join(out_dir, "runs", f"run_{cand_id}_discarded.json"), "w", encoding="utf-8") as fh:
                    json.dump({"reason": "thin_bins", **row}, fh, indent=2, default=_json_default)
            except Exception:
                pass
            cand_id += 1
            continue

        rows.append(row)
        kept += 1
        try:
            with open(os.path.join(out_dir, "runs", f"run_{cand_id}.json"), "w", encoding="utf-8") as fh:
                json.dump(row, fh, indent=2, default=_json_default)
            info = row.get("panel_info") if "panel_info" in row else _result_to_panel_info(DidStudy(cfg).run())
            with open(os.path.join(out_dir, "runs", f"panel_{cand_id}.json"), "w", encoding="utf-8") as fh2:
                json.dump(_select_panel_info_fields(info), fh2, indent=2, default=_json_default)
        except Exception:
            pass

        cand_id += 1

    results_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    # Ensure expected columns exist before downstream ops
    required_cols = [
        "coef", "se", "p", "p_wcb", "pta_p", "wcb_joint_p", "n_clusters", "mde",
        "n_bins", "post_share", "thin_bins", "honest_pass", "config_snapshot",
    ]
    for c in required_cols:
        if c not in results_df.columns:
            results_df[c] = np.nan

    results_df["meets_target_mde"] = results_df["mde"].apply(lambda x: (float(x) <= float(target_mde)) if np.isfinite(x) else False)
    results_df["mde_ratio"] = results_df.apply(
        lambda r: (abs(float(r.get("coef", np.nan))) / float(r.get("mde", np.nan))) if (np.isfinite(r.get("coef")) and np.isfinite(r.get("mde")) and float(r.get("mde")) > 0) else np.nan,
        axis=1,
    )

    # Sort for reading
    def _nan_high(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return float("inf")

    results_df = results_df.sort_values(
        by=[
            "meets_target_mde",
            "p",
            "p_wcb",
            "pta_p",
            "wcb_joint_p",
            "honest_pass",
            "mde",
            "se",
        ],
        ascending=[False, True, True, True, True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    meeting_df = results_df[results_df["meets_target_mde"] == True]  # noqa: E712
    highlight_df = meeting_df[(meeting_df["p"].apply(lambda x: np.isfinite(x) and x < alpha)) | (meeting_df["p_wcb"].apply(lambda x: np.isfinite(x) and x < alpha))]
    pta_wcb_df = meeting_df[(meeting_df["pta_p"].apply(lambda x: np.isfinite(x) and x < alpha)) & (meeting_df["wcb_joint_p"].apply(lambda x: np.isfinite(x) and x < alpha))]
    robust_df = meeting_df[meeting_df["honest_pass"] == True]  # noqa: E712

    best_overall = results_df.iloc[0].to_dict() if not results_df.empty else {}

    summary = {
        "out_dir": out_dir,
        "meeting_count": int(meeting_df.shape[0]),
        "highlight_count": int(highlight_df.shape[0]),
        "pta_wcb_count": int(pta_wcb_df.shape[0]),
        "robust_count": int(robust_df.shape[0]),
        "meeting_ids": meeting_df.index.tolist(),
        "highlight_ids": highlight_df.index.tolist(),
        "pta_wcb_ids": pta_wcb_df.index.tolist(),
        "robust_ids": robust_df.index.tolist(),
        "best_overall": best_overall,
    }

    try:
        results_df.to_csv(os.path.join(out_dir, "results_df.csv"), index=True)
        meeting_df.to_csv(os.path.join(out_dir, "meeting_configs.csv"), index=True)
        highlight_df.to_csv(os.path.join(out_dir, "highlight_configs.csv"), index=True)
        pta_wcb_df.to_csv(os.path.join(out_dir, "pta_wcb_configs.csv"), index=True)
        robust_df.to_csv(os.path.join(out_dir, "robust_configs.csv"), index=True)
        with open(os.path.join(out_dir, "winner.json"), "w", encoding="utf-8") as fh:
            json.dump(best_overall, fh, indent=2, default=_json_default)
    except Exception:
        pass

    return results_df, summary

def analyze_results(results_df: pd.DataFrame, out_dir: str) -> Dict[str, Any]:
    import statsmodels.formula.api as smf  # type: ignore
    df = results_df.copy()
    out: Dict[str, Any] = {"notes": []}
    if df.empty:
        return out

    # Flatten snapshot
    snap = df["config_snapshot"].apply(lambda d: d if isinstance(d, dict) else {})
    df["use_log_outcome"] = snap.apply(lambda s: bool(s.get("use_log_outcome", False)))
    df["differenced"] = snap.apply(lambda s: bool(s.get("differenced", False)))
    df["use_lag_levels_in_diff"] = snap.apply(lambda s: bool(s.get("use_lag_levels_in_diff", False)))
    df["min_pre"] = snap.apply(lambda s: int(s.get("min_pre", 0)))
    df["min_post"] = snap.apply(lambda s: int(s.get("min_post", 0)))
    df["supdem_mode"] = snap.apply(lambda s: str(s.get("supdem_mode", "sum")))
    df["covar_set_size"] = df["covariates_effective"].apply(lambda s: 0 if (s is None or s == "") else len(str(s).split(",")))

    safe = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mde", "se"])  # minimal cleaning
    if safe.empty:
        return out

    f_mde = (
        "mde ~ C(use_log_outcome) + C(differenced) + C(use_lag_levels_in_diff) + min_pre + min_post + "
        "C(supdem_mode) + n_clusters + n_bins + covar_set_size + post_share"
    )
    f_se = (
        "se ~ C(use_log_outcome) + C(differenced) + C(use_lag_levels_in_diff) + min_pre + min_post + "
        "C(supdem_mode) + n_clusters + n_bins + covar_set_size + post_share"
    )
    try:
        m_mde = smf.ols(f_mde, data=safe).fit(cov_type="HC1")
        m_se = smf.ols(f_se, data=safe).fit(cov_type="HC1")
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({"coef": m_mde.params, "se_robust": m_mde.bse, "t": m_mde.tvalues, "p": m_mde.pvalues}).to_csv(
            os.path.join(out_dir, "drivers_ols_summary_mde.csv")
        )
        pd.DataFrame({"coef": m_se.params, "se_robust": m_se.bse, "t": m_se.tvalues, "p": m_se.pvalues}).to_csv(
            os.path.join(out_dir, "drivers_ols_summary_se.csv")
        )
        out["top_mde_drivers"] = [
            f"{ix}: coef={row.coef:.4g}, p={row.p:.3g}" for ix, row in pd.DataFrame({
                "coef": m_mde.params, "p": m_mde.pvalues
            }).sort_values("p").head(5).iterrows()
        ]
    except Exception as e:
        out["notes"].append(f"Drivers OLS failed: {e}")
    return out

def write_report(results_df: pd.DataFrame, summary: Dict[str, Any], target_mde: float, out_dir: str, alpha: float = 0.05) -> None:
    lines: List[str] = []
    lines.append("# CCUS-informed Power Tuner (2–3 bins, VIF)")
    lines.append(f"Target MDE: {target_mde:.6g}; alpha={alpha}")
    lines.append("MDE formula: (t_{1−α/2, G−1} + t_{power, G−1}) × SE")

    meeting = results_df[results_df["meets_target_mde"] == True]  # noqa: E712
    robust = meeting[(meeting["pta_p"].apply(lambda x: np.isfinite(x) and x < alpha)) & (meeting["wcb_joint_p"].apply(lambda x: np.isfinite(x) and x < alpha)) & (meeting["honest_pass"] == True)]  # noqa: E712

    lines.append("\n## Executive Summary")
    if not robust.empty:
        lines.append("Yes — at least one configuration achieves MDE < |effect| and is robust (PTA, WCB pretrend, HonestDiD).")
    elif not meeting.empty:
        lines.append("Partially — some configs achieve MDE < |effect| but fail at least one robustness check.")
    else:
        lines.append("No — no configuration reaches MDE < |effect|.")

    lines.append(\
        f"Counts — meeting: {int(meeting.shape[0])}, highlighted (significant): {int(((meeting['p']<alpha) | (meeting['p_wcb']<alpha)).sum())}, PTA+WCB clean: {int(((meeting['pta_p']<alpha) & (meeting['wcb_joint_p']<alpha)).sum())}, HonestDiD-pass: {int((meeting['honest_pass']==True).sum())}."  # noqa: E712
    )

    lines.append("\n## Top Configurations")
    top = results_df.head(10).copy()
    for _, row in top.iterrows():
        snap = row.get("config_snapshot", {}) or {}
        dq = snap.get("dose_quantiles")
        flags = []
        if (np.isfinite(row.get("p")) and float(row.get("p")) < alpha) or (np.isfinite(row.get("p_wcb")) and float(row.get("p_wcb")) < alpha):
            flags.append("★ sig")
        if np.isfinite(row.get("pta_p")) and float(row.get("pta_p")) < alpha:
            flags.append("✓ PTA")
        if np.isfinite(row.get("wcb_joint_p")) and float(row.get("wcb_joint_p")) < alpha:
            flags.append("Ⓦ WCB-pre")
        if bool(row.get("honest_pass")):
            flags.append("✔ HonestDiD")
        lines.append(
            "- "
            + f"coef={row.get('coef'):.4g}, se={row.get('se'):.4g}, mde={row.get('mde'):.4g}, mde_ratio={row.get('mde_ratio'):.4g}; "
            + f"p={row.get('p')}, p_wcb={row.get('p_wcb')}, pta_p={row.get('pta_p')}, wcb_pre={row.get('wcb_joint_p')}; "
            + f"n_clusters={row.get('n_clusters')}, post_share={row.get('post_share')}; dose_quantiles={dq}, n_bins={int(row.get('n_bins', 0))} "
            + ("[" + ", ".join(flags) + "]" if flags else "")
        )

    lines.append("\n## Drivers (summary)")
    try:
        drivers = analyze_results(results_df, out_dir)
        for s in drivers.get("top_mde_drivers", [])[:5]:
            lines.append(f"- {s}")
    except Exception:
        lines.append("- Drivers analysis unavailable")

    report = "\n".join(lines)
    # Diagnostics on MDE audit
    try:
        if (results_df.get("mde_audit") == "differs_from_theory").any():
            report += "\n\nNote: Tuner used theory-correct MDE (two-sided t, df=G−1); base study MDE differs."
    except Exception:
        pass
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "power_tuner_report.md"), "w", encoding="utf-8") as fh:
            fh.write(report)
    except Exception:
        pass
    # Print report with a fallback for Windows consoles that lack UTF-8
    try:
        print(report)
    except Exception:
        try:
            print(report.encode("cp1252", errors="ignore").decode("cp1252", errors="ignore"))
        except Exception:
            print(report.encode("ascii", errors="ignore").decode("ascii", errors="ignore"))


# ==============================================================================
# Parallel runner (process-based, tqdm progress)
# ==============================================================================

_PT_GLOBAL: Dict[str, Any] = {"df": None, "small_pos": None}


def _pt_init_worker(data_path: str) -> None:
    import pandas as _pd
    try:
        df = _pd.read_csv(data_path)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to read data in worker: {e}")
    _PT_GLOBAL["df"] = df
    # robust small_pos from eor_capacity column if present
    sp = 0.0
    try:
        if "eor_capacity" in df.columns:
            vals = _pd.to_numeric(df["eor_capacity"], errors="coerce")
            vals = vals[vals > 0]
            if len(vals) > 0:
                sp = float(vals.median()) * 1e-6
    except Exception:
        sp = 0.0
    _PT_GLOBAL["small_pos"] = sp


def _pt_worker_eval(
    idx: int,
    overrides: Dict[str, Any],
    base_kwargs: Dict[str, Any],
    alpha: float,
    power_target: float,
    use_wcb: bool,
    honestdid: bool,
) -> Tuple[int, Dict[str, Any]]:
    df = _PT_GLOBAL.get("df")
    if df is None:
        raise RuntimeError("Worker not initialised with data frame.")
    cfg = StudyConfig(df=df, **base_kwargs)
    cfg.use_wcb = bool(use_wcb)
    cfg.honestdid_enable = bool(honestdid)
    # apply overrides
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            continue
        if k == "treat_threshold" and v == "small_pos":
            setattr(cfg, k, float(_PT_GLOBAL.get("small_pos") or 0.0))
        else:
            setattr(cfg, k, v)
    dq = overrides.get("dose_quantiles") or []
    cfg.n_bins = (len(dq) - 1) if dq else getattr(cfg, "n_bins", None)

    row = evaluate_config(cfg, alpha=alpha, power_target=power_target)
    row["_candidate_index"] = idx
    return idx, row


def run_search_parallel_from_path(
    data_path: str,
    base_config_kwargs: Dict[str, Any],
    target_mde: float,
    *,
    threads: int = 4,
    alpha: float = 0.05,
    power_target: float = 0.80,
    max_candidates: int = 160,
    use_wcb: bool = False,
    honestdid: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parallel search using processes; tracks progress with tqdm and combines results.
    base_config_kwargs should contain any StudyConfig kwargs except df (e.g., artifact_dir, covariates).
    """
    # Template cfg for search-space discovery
    tmpl = StudyConfig(df=pd.DataFrame(), **{k: v for k, v in base_config_kwargs.items() if k != "df"})
    grid = make_search_space(tmpl)

    # Build candidate overrides
    keys = [
        k for k in [
            "use_log_outcome",
            "differenced",
            "use_lag_levels_in_diff",
            "outcome_mode",
            "supdem_mode",
            "min_pre",
            "min_post",
            "treat_threshold",
            "dose_quantiles",
            "covariates",
        ] if k in grid
    ]
    vals = [grid[k] for k in keys]
    from itertools import product as _product
    cand_list: List[Dict[str, Any]] = []
    for combo in _product(*vals):
        overrides = dict(zip(keys, combo))
        dq = overrides.get("dose_quantiles") or []
        n_bins = len(dq) - 1 if dq else 0
        if n_bins in (2, 3):
            cand_list.append(overrides)
        if len(cand_list) >= max_candidates:
            break

    # Output directory
    root = base_config_kwargs.get("artifact_dir") or "./_artifacts"
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, f"power_search_{ts}")
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)

    # Parallel pool
    from concurrent.futures import ProcessPoolExecutor, as_completed
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        def tqdm(x, total=None):
            return x

    rows: List[Dict[str, Any]] = []
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, int(threads)), initializer=_pt_init_worker, initargs=(data_path,)) as ex:
        for i, ov in enumerate(cand_list):
            futures.append(ex.submit(_pt_worker_eval, i, ov, base_config_kwargs, alpha, power_target, use_wcb, honestdid))

        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                idx, row = fut.result()
            except Exception as e:  # noqa: BLE001
                row = {"error": str(e), "_candidate_index": -1}
            rows.append(row)
            # Per-run persistence
            try:
                ridx = row.get("_candidate_index", len(rows)-1)
                with open(os.path.join(out_dir, "runs", f"run_{ridx}.json"), "w", encoding="utf-8") as fh:
                    json.dump(row, fh, indent=2, default=_json_default)
                info = row.get("panel_info")
                if info is not None:
                    with open(os.path.join(out_dir, "runs", f"panel_{ridx}.json"), "w", encoding="utf-8") as fh2:
                        json.dump(info, fh2, indent=2, default=_json_default)
            except Exception:
                pass

    # Combine
    results_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if results_df.empty:
        results_df = pd.DataFrame(columns=[
            "coef", "se", "p", "p_wcb", "pta_p", "wcb_joint_p", "n_clusters", "mde",
            "n_bins", "post_share", "thin_bins", "honest_pass", "config_snapshot",
        ])

    # Derived metrics and sorting like run_search
    results_df["meets_target_mde"] = results_df["mde"].apply(lambda x: (float(x) <= float(target_mde)) if np.isfinite(x) else False)
    results_df["mde_ratio"] = results_df.apply(
        lambda r: (abs(float(r.get("coef", np.nan))) / float(r.get("mde", np.nan))) if (np.isfinite(r.get("coef")) and np.isfinite(r.get("mde")) and float(r.get("mde")) > 0) else np.nan,
        axis=1,
    )

    results_df = results_df.sort_values(
        by=[
            "meets_target_mde",
            "p",
            "p_wcb",
            "pta_p",
            "wcb_joint_p",
            "honest_pass",
            "mde",
            "se",
        ],
        ascending=[False, True, True, True, True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    meeting_df = results_df[results_df["meets_target_mde"] == True]  # noqa: E712
    highlight_df = meeting_df[(meeting_df["p"].apply(lambda x: np.isfinite(x) and x < alpha)) | (meeting_df["p_wcb"].apply(lambda x: np.isfinite(x) and x < alpha))]
    pta_wcb_df = meeting_df[(meeting_df["pta_p"].apply(lambda x: np.isfinite(x) and x < alpha)) & (meeting_df["wcb_joint_p"].apply(lambda x: np.isfinite(x) and x < alpha))]
    robust_df = meeting_df[meeting_df["honest_pass"] == True]  # noqa: E712

    best_overall = results_df.iloc[0].to_dict() if not results_df.empty else {}

    summary = {
        "out_dir": out_dir,
        "meeting_count": int(meeting_df.shape[0]),
        "highlight_count": int(highlight_df.shape[0]),
        "pta_wcb_count": int(pta_wcb_df.shape[0]),
        "robust_count": int(robust_df.shape[0]),
        "meeting_ids": meeting_df.index.tolist(),
        "highlight_ids": highlight_df.index.tolist(),
        "pta_wcb_ids": pta_wcb_df.index.tolist(),
        "robust_ids": robust_df.index.tolist(),
        "best_overall": best_overall,
    }

    # Persist CSVs and winner
    try:
        results_df.to_csv(os.path.join(out_dir, "results_df.csv"), index=True)
        meeting_df.to_csv(os.path.join(out_dir, "meeting_configs.csv"), index=True)
        highlight_df.to_csv(os.path.join(out_dir, "highlight_configs.csv"), index=True)
        pta_wcb_df.to_csv(os.path.join(out_dir, "pta_wcb_configs.csv"), index=True)
        robust_df.to_csv(os.path.join(out_dir, "robust_configs.csv"), index=True)
        with open(os.path.join(out_dir, "winner.json"), "w", encoding="utf-8") as fh:
            json.dump(best_overall, fh, indent=2, default=_json_default)
    except Exception:
        pass

    return results_df, summary
