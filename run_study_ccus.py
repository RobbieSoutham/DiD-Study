"""
Run the CCUS DiD study end-to-end using the checked-in pipeline.

Usage (ensuring the new_venv environment is used):
    new_venv\\Scripts\\python run_study_ccus.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _ensure_workspace_julia_on_path() -> None:
    """Expose the workspace-local Julia install used by fwildclusterboot."""
    here = Path(__file__).resolve()
    for root in (here.parent, *here.parents):
        julia_bin = root / "tools" / "juliaup" / "bin"
        if (julia_bin / "julia").exists():
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            julia_path = str(julia_bin)
            if julia_path not in path_parts:
                os.environ["PATH"] = julia_path + os.pathsep + os.environ.get("PATH", "")
            return


_ensure_workspace_julia_on_path()

from did_study.helpers.config import StudyConfig
from did_study.helpers.defaults import DEFAULT_MAPPING
from did_study.reporting.plotting import (
    plot_differences_event_agg,
    plot_dose_distribution,
    plot_differences_att_gt,
    plot_event_study_line,
    plot_honest_did_M_curve,
    plot_mean_dose_by_tau,
    plot_support_by_tau,
    PlotTheme,
)
from did_study.study import DidStudy

dose_bins = [0, 1.0, 5.0, np.inf]
bin_labels = ['Small', 'Medium', 'Large']
RNG_SEED = 123
PLACEBO_DRAWS = int(os.environ.get("CCUS_PLACEBO_DRAWS", "1000"))
RUN_LOO = os.environ.get("CCUS_SKIP_LOO", "0") != "1"
USE_CACHE = os.environ.get("CCUS_USE_CACHE", "0") == "1"
RUN_WCB = os.environ.get("CCUS_SKIP_WCB", "0") != "1"
RUN_HONESTDID = os.environ.get("CCUS_SKIP_HONESTDID", "0") != "1"


def _ensure_dirs() -> None:
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/data").mkdir(parents=True, exist_ok=True)
    Path("results/tex").mkdir(parents=True, exist_ok=True)


def _add_ci(df_es: pd.DataFrame) -> pd.DataFrame:
    """Add 95% CI columns if only beta/se are present."""
    out = df_es.copy()
    if {"beta", "se"}.issubset(out.columns):
        out["lo"] = out["beta"] - 1.96 * out["se"]
        out["hi"] = out["beta"] + 1.96 * out["se"]
    return out


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _build_config(raw_df: pd.DataFrame, *, honestdid_enable: bool) -> StudyConfig:
    covariates = [
        "Demand_heat",
        "Demand_renewables_and_waste",
        "energy_demand_fossil_fuels",
        "CPI_growth",
        "GDP_per_capita_PPP",
    ]

    return StudyConfig(
        df=raw_df,
        outcome_mode="direct",
        supdem_mode="direct",
        mapping=DEFAULT_MAPPING,
        unit_cols=("Country", "CCUS_sector"),
        year_col="Year",
        outcome_col="Emissions",
        emissions_sector_col="emissions_sector",
        capacity_col="eor_capacity",
        sector_col="Sector",
        covariates=covariates,
        differenced=False,
        use_log_outcome=True,
        use_lag_levels_in_diff=False,
        min_pre=4,
        min_post=4,
        n_bins=len(dose_bins) - 1,
        dose_bins=dose_bins,
        dose_bins_right_closed=False,
        treat_threshold=0,
        pre=5,
        post=10,
        cluster_col="unit_id",
        use_wcb=RUN_WCB,
        min_cluster_support=2,
        honestdid_enable=honestdid_enable and RUN_HONESTDID,
        seed=RNG_SEED,
    )


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _normal_mde_from_se(
    se: Any,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """MDE using the normal critical values stated in the manuscript."""
    se_float = _as_float(se)
    if not np.isfinite(se_float):
        return float("nan")
    z = NormalDist().inv_cdf
    alpha_crit = z(1 - alpha / 2) if two_sided else z(1 - alpha)
    power_crit = z(power)
    return float(abs(se_float) * (alpha_crit + power_crit))


def _clean_unit_label(label: Any) -> str:
    return str(label).replace("_", "--", 1)


def _treated_unit_table(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    treated = panel_df.loc[panel_df["treated_ever"] == 1].copy()
    for unit_id, dfu in treated.groupby("unit_id", sort=False):
        g = int(dfu["g"].dropna().iloc[0])
        at_g = dfu.loc[dfu["Year"] == g]
        initial_capacity = (
            float(at_g["dose_level"].iloc[0]) if not at_g.empty else float("nan")
        )
        rows.append(
            {
                "unit_id": unit_id,
                "country_sector": _clean_unit_label(unit_id),
                "adoption_year": g,
                "initial_capacity_mtco2yr": initial_capacity,
                "pre_years": int((dfu["Year"] < g).sum()),
                "post_years": int((dfu["Year"] >= g).sum()),
                "max_supported_tau": int(dfu["event_time"].dropna().max()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["adoption_year", "country_sector"])
    out.to_csv("results/data/treated_units_revision_support.csv", index=False)
    return out


def _plot_loo(loo_df: pd.DataFrame) -> None:
    if loo_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(loo_df))
    ax.axvline(0, color="0.45", linewidth=1)
    ax.errorbar(
        loo_df["att_overall"],
        y,
        xerr=[
            loo_df["att_overall"] - loo_df["ci_low"],
            loo_df["ci_high"] - loo_df["att_overall"],
        ],
        fmt="o",
        color="#1f4e79",
        ecolor="#6d8fb3",
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(loo_df["dropped_unit"])
    ax.invert_yaxis()
    ax.set_xlabel(r"ATT$^{\circ}$ after dropping one treated unit")
    ax.set_ylabel("Dropped treated unit")
    fig.tight_layout()
    fig.savefig("results/figures/leave_one_treated_unit_out.png", dpi=220)
    plt.close(fig)


def _run_leave_one_out(raw_df: pd.DataFrame, treated_units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if treated_units.empty:
        return pd.DataFrame()

    for _, treated_row in treated_units.iterrows():
        unit_id = treated_row["unit_id"]
        if "_" not in str(unit_id):
            continue
        country, sector = str(unit_id).split("_", 1)
        print(f"       [LOO] dropping treated unit: {country} -- {sector}")
        df_loo = raw_df.copy(deep=False)
        mask = (df_loo["Country"] == country) & (df_loo["CCUS_sector"] == sector)
        df_loo["eor_capacity"] = raw_df["eor_capacity"].mask(mask, 0.0)
        cfg_loo = _build_config(df_loo, honestdid_enable=False)
        result = DidStudy(cfg_loo).run(run_event_study=False, run_honest_did=False)
        att_test = getattr(result, "att_test", None)
        att = _as_float(_get(att_test, "att_overall"))
        se = _as_float(_get(att_test, "se_overall"))
        rows.append(
            {
                "dropped_unit": _clean_unit_label(unit_id),
                "att_overall": att,
                "se_overall": se,
                "p_overall": _as_float(_get(att_test, "p_overall")),
                "p_boot": _as_float(_get(att_test, "p_boot")),
                "ci_low": att - 1.96 * se if np.isfinite(se) else float("nan"),
                "ci_high": att + 1.96 * se if np.isfinite(se) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv("results/data/loo.csv", index=False)
    _plot_loo(out)
    return out


def _within_twfe_coef(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str = "log_y",
    unit_col: str = "unit_id",
    year_col: str = "Year",
) -> float:
    use = df[[outcome_col, treatment_col, unit_col, year_col]].dropna().copy()
    if use[treatment_col].var() == 0:
        return float("nan")
    y = use[outcome_col].astype(float)
    d = use[treatment_col].astype(float)
    y_dm = y - y.groupby(use[unit_col]).transform("mean") - y.groupby(use[year_col]).transform("mean") + y.mean()
    d_dm = d - d.groupby(use[unit_col]).transform("mean") - d.groupby(use[year_col]).transform("mean") + d.mean()
    denom = float(np.dot(d_dm, d_dm))
    if denom == 0:
        return float("nan")
    return float(np.dot(d_dm, y_dm) / denom)


def _run_placebo_twfe(panel_df: pd.DataFrame, draws: int = PLACEBO_DRAWS) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    base = panel_df.copy()
    treated = (
        base.loc[base["treated_ever"] == 1, ["unit_id", "g"]]
        .drop_duplicates()
        .dropna()
    )
    control_units = sorted(base.loc[base["treated_ever"] == 0, "unit_id"].dropna().unique())
    years = treated["g"].astype(int).to_numpy()
    if len(treated) == 0 or len(control_units) < len(treated):
        return pd.DataFrame()

    observed = _within_twfe_coef(base.assign(placebo_treat=base["post"]), treatment_col="placebo_treat")
    rows = [{"draw": -1, "att_twfe": observed, "kind": "observed"}]

    for i in range(draws):
        selected = rng.choice(control_units, size=len(treated), replace=False)
        assigned_years = rng.choice(years, size=len(treated), replace=True)
        g_map = dict(zip(selected, assigned_years))
        placebo = base["unit_id"].map(g_map)
        d = ((placebo.notna()) & (base["Year"].astype(int) >= placebo.astype("float"))).astype(int)
        coef = _within_twfe_coef(base.assign(placebo_treat=d), treatment_col="placebo_treat")
        rows.append({"draw": i, "att_twfe": coef, "kind": "placebo"})

    out = pd.DataFrame(rows)
    observed_rank = float((out.loc[out["kind"] == "placebo", "att_twfe"] >= observed).mean())
    out["observed_upper_tail_share"] = observed_rank
    out.to_csv("results/data/placebo_twfe_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(out.loc[out["kind"] == "placebo", "att_twfe"].dropna(), bins=35, color="#b8c7d9", edgecolor="white")
    ax.axvline(observed, color="#9b2f2f", linewidth=2, label=f"Observed TWFE diagnostic = {observed:.3f}")
    ax.set_xlabel("Placebo fixed-effects coefficient")
    ax.set_ylabel("Draws")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("results/figures/placebo_twfe_distribution.png", dpi=220)
    plt.close(fig)
    return out


def _normalise_event_agg(event_agg: Any) -> pd.DataFrame:
    df = event_agg.reset_index() if hasattr(event_agg, "reset_index") else pd.DataFrame(event_agg)
    df.columns = [
        "_".join(str(x) for x in col if str(x) != "") if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    tau_col = next((c for c in df.columns if c in {"event_time", "relative_period", "rel_period", "time", "index"}), df.columns[0])
    att_col = next((c for c in df.columns if "ATT" in c or "att" in c or "estimate" in c.lower()), None)
    se_col = next((c for c in df.columns if "std_error" in c.lower() or c.lower().endswith("_se") or c.lower() == "se"), None)
    out = pd.DataFrame({"event_time": pd.to_numeric(df[tau_col], errors="coerce")})
    if att_col:
        out["att"] = pd.to_numeric(df[att_col], errors="coerce")
    if se_col:
        out["se"] = pd.to_numeric(df[se_col], errors="coerce")
    out = out.dropna(subset=["event_time"]).sort_values("event_time")
    return out


def _write_alternative_aggregation(event_df: pd.DataFrame) -> pd.DataFrame:
    post = event_df.loc[event_df["event_time"] >= 0].copy()
    rows = []
    windows = {
        "full_tau_0_10": post.loc[post["event_time"].between(0, 10)],
        "early_tau_0_5": post.loc[post["event_time"].between(0, 5)],
        "late_tau_6_10": post.loc[post["event_time"].between(6, 10)],
        "trimmed_tau_0_7": post.loc[post["event_time"].between(0, 7)],
    }
    for name, dfw in windows.items():
        rows.append(
            {
                "window": name,
                "min_tau": int(dfw["event_time"].min()) if not dfw.empty else None,
                "max_tau": int(dfw["event_time"].max()) if not dfw.empty else None,
                "n_event_times": int(len(dfw)),
                "mean_att": float(dfw["att"].mean()) if "att" in dfw and not dfw.empty else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv("results/data/alternative_event_time_aggregations.csv", index=False)
    return out


def _unwrap_results(res: Any) -> Dict[str, Any]:
    """
    Normalize either DidStudyResult or StudyPrintable-like objects into a dict
    with common keys used downstream.
    """
    out: Dict[str, Any] = {}

    # Panel
    panel_df = getattr(getattr(res, "data", None), "panel", None)
    if panel_df is None:
        panel_df = getattr(res, "panel_df", None)
    out["panel_df"] = panel_df

    # Headline ATT differences
    if hasattr(res, "att_test"):
        out["att_test"] = res.att_test

    # Event study
    if hasattr(res, "event_study"):
        out["event_study"] = res.event_study
    elif hasattr(res, "es_pooled"):
        out["event_study"] = SimpleNamespace(coefs=getattr(res, "es_pooled", None), wcb_p=getattr(res, "es_wcb_p", None))

    # Support by tau
    if hasattr(res, "support_by_tau"):
        out["support_by_tau"] = res.support_by_tau

    # HonestDiD
    if hasattr(res, "honest_did"):
        out["honest_did"] = res.honest_did

    # WCB meta / es_wcb_p
    out["es_wcb_p"] = getattr(res, "es_wcb_p", None)
    out["wcb_meta"] = getattr(res, "wcb_meta", {})

    return out


def main() -> None:
    _ensure_dirs()

    print("[1/4] Loading raw panel ds_no_prep.csv ...")
    raw_df = pd.read_csv("ds_no_prep.csv")
    print(f"       Loaded {len(raw_df):,} rows.")

    cfg = _build_config(raw_df, honestdid_enable=True)
    covariates = list(cfg.covariates or [])
    sns.set_theme()
    # Try to load cached study only when explicitly requested. Revision runs are
    # fresh by default so diagnostics cannot silently use stale outputs.
    use_cache = USE_CACHE or os.environ.get("DID_STUDY_USE_CACHE") == "1"
    cache_path = Path("lateststudys.pkl")
    results = None
    if use_cache and cache_path.exists():
        try:
            results = pickle.load(cache_path.open("rb"))
            print(f"[2/4] Loaded cached study from {cache_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[2/4] Failed to load cache ({exc}); running study ...")
            cache_path.unlink(missing_ok=True)

    if results is None:
        print("[2/4] Running DidStudy ... (this can take a while)")
        study = DidStudy(cfg)
        results = study.run()
        print("       Study run complete.")
        if use_cache:
            cache_tmp = cache_path.with_suffix(".tmp")
            try:
                with cache_tmp.open("wb") as fh:
                    pickle.dump(results, fh)
                cache_tmp.replace(cache_path)
                print(f"       Saved study to {cache_path}")
            except Exception as exc:  # noqa: BLE001
                cache_tmp.unlink(missing_ok=True)
                cache_path.unlink(missing_ok=True)
                print(f"[warn] Could not save study cache: {exc}")

    res_norm = _unwrap_results(results)
    panel_df = res_norm.get("panel_df")
    att_test_obj = res_norm.get("att_test")

    print("[3/4] Saving figures ...")
    # Event-study line plot
    es_obj = res_norm.get("event_study")
    if es_obj is not None and getattr(es_obj, "coefs", None) is not None and not es_obj.coefs.empty:
        df_es = _add_ci(es_obj.coefs)
        plot_event_study_line(
            df_es,
            title=None,
            xlabel=r"Event time ($\tau$)",
            ylabel="Effect on log emissions",
            ref_tau=-1,
            save="results/figures/event_study.png",
            show=False,
        )
        df_es.to_csv("results/data/event_study_coefs.csv", index=False)
    else:
        print("       [warn] No event-study results to plot.")

    # Differences-in-differences ATT(g,t) plot
    res = att_test_obj.attgt_obj.aggregate(
        type_of_aggregation="event",
        boot_iterations=9999
        )
    event_agg_df = _normalise_event_agg(res)
    event_agg_df.to_csv("results/data/differences_event_agg.csv", index=False)
    _write_alternative_aggregation(event_agg_df)
    plot_differences_event_agg(
        res,
        title=None,
        xlabel=r"Event time ($\tau$)",
        ylabel="ATT(g, t)",
        save="results/figures/differences_att_gt.png",
        show=False,
    )
    #att_gt_df.to_csv("results/data/differences_att_gt.csv", index=False)

    # Support by tau (leads only)
    if panel_df is not None and {"event_time", "unit_id"}.issubset(panel_df.columns):
        support = (
            panel_df.dropna(subset=["event_time"])
            .groupby("event_time")["unit_id"]
            .nunique()
            .reset_index(name="units")
        )
        plot_support_by_tau(
            support,
            title=None,
            xlabel=r"Event time ($\tau$)",
            ylabel="Number of treated units",
            save="results/figures/support_by_tau.png",
            show=False,
        )
        support.to_csv("results/data/support_by_tau.csv", index=False)

    revision_treated_units = pd.DataFrame()
    if panel_df is not None and {"treated_ever", "unit_id", "g", "Year", "dose_level"}.issubset(panel_df.columns):
        revision_treated_units = _treated_unit_table(panel_df)
        if RUN_LOO:
            print("       Running leave-one-treated-unit-out robustness checks ...")
            _run_leave_one_out(raw_df, revision_treated_units)
        else:
            print("       Skipping LOO checks because CCUS_SKIP_LOO=1.")
        print(f"       Running {PLACEBO_DRAWS} placebo timing diagnostics ...")
        _run_placebo_twfe(panel_df, PLACEBO_DRAWS)
    
    # Dose distribution histogram: repeated initial capacity over treated unit-years.
    dose_series = None
    if panel_df is not None and {"unit_id", "Year", "g", "treated_now", "dose_level"}.issubset(panel_df.columns):
        init = (
            panel_df.loc[panel_df["Year"] == panel_df["g"]]
            .groupby("unit_id")["dose_level"]
            .first()
        )
        dose_series = (
            panel_df.loc[panel_df["treated_now"] == 1, "unit_id"]
            .map(init)
            .dropna()
            .astype(float)
        )
    elif isinstance(getattr(getattr(results, "data", None), "info", None), dict):
        dose_series = getattr(results.data, "info", {}).get("dose_series", None)
    if dose_series is not None and len(dose_series) > 0:
        plot_dose_distribution(
            dose_series,
            bin_edges=dose_bins,
            bin_labels = bin_labels,
            bins="auto",
            title=None,
            xlabel="Initial CCUS capacity (Mt CO$_2$/yr)",
            ylabel="Number of Units",
            save="results/figures/dose_distribution.png",
            show=False,
        )

    # HonestDiD sensitivity
    hd = res_norm.get("honest_did")
    if hd:
        plot_honest_did_M_curve(
            hd.get("M", []),
            hd.get("lo", []),
            hd.get("hi", []),
            hd.get("theta_hat", None),
            title=None,
            xlabel=r"$M$ (relative magnitude bound)",
            ylabel="Bounded average effect",
            save="results/figures/honest_did.png",
            show=False,
        )
        Path("results/data").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "M": hd.get("M", []),
                "lo": hd.get("lo", []),
                "hi": hd.get("hi", []),
            }
        ).to_csv("results/data/honest_did_bounds.csv", index=False)

    # Mean dose by event time (pooled)
    dfp = panel_df.copy()

# 1) Ensure adoption year g exists
    if "g" not in dfp.columns:
        g = dfp.loc[dfp["dose_level"] > 0].groupby("unit_id")["Year"].min()
        dfp["g"] = dfp["unit_id"].map(g)

    # 2) Event time for *all* years of treated units
    dfp["event_time"] = np.where(dfp["g"].notna(), dfp["Year"] - dfp["g"], np.nan)

    # 3) Unit-level bin from initial dose at adoption (year == g)
    init = dfp.loc[dfp["Year"] == dfp["g"]].groupby("unit_id")["dose_level"].first()
    unit_bin = pd.cut(init, [-np.inf, 1, 5, np.inf],
                    labels=["Small", "Medium", "Large"], right=False)
    dfp["dose_bin"] = dfp["unit_id"].map(unit_bin)

    # 4) Use the *time-varying* dose
    dfp["dose_mt"] = dfp["dose_level"].astype(float)
    dfp = dfp[dfp["event_time"].between(-10, 40)]

    if panel_df is not None and {"event_time", "dose_level"}.issubset(panel_df.columns):
        try:
            plot_mean_dose_by_tau(
                dfp,
                title=None,
                xlabel=r"Event time ($\tau$)",
                ylabel="Mean dose (Mt CO$_2$/yr)",
                show_pooled=True,
                #group_labels=["Pooled"] + bin_labels,
                legend_title="Dose bin",
                save="results/figures/mean_dose_by_tau.png",
                show=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] mean dose by tau plot failed: {exc}")

    print("[4/4] Writing summary JSON ...")
    att_used = getattr(att_test_obj, "used", None)
    att_n = int(len(att_used)) if isinstance(att_used, pd.DataFrame) else None
    att_clusters = None
    if isinstance(att_used, pd.DataFrame):
        if "unit_id" in att_used.columns:
            att_clusters = int(att_used["unit_id"].nunique())
        elif isinstance(att_used.index, pd.MultiIndex):
            level_name = "unit" if "unit" in att_used.index.names else att_used.index.names[0]
            att_clusters = int(att_used.index.get_level_values(level_name).nunique())
    att_mde = _normal_mde_from_se(_get(att_test_obj, "se_overall"))
    summary = {
        "att_test": {
            "att_overall": _get(att_test_obj, "att_overall"),
            "se_overall": _get(att_test_obj, "se_overall"),
            "p_overall": _get(att_test_obj, "p_overall"),
            "p_boot": _get(att_test_obj, "p_boot"),
            "mde": att_mde,
            "mde_alpha": 0.05,
            "mde_power": 0.80,
            "mde_method": "normal_approximation",
            "clusters": att_clusters,
            "n": att_n,
            "covariates_used": _get(att_test_obj, "covariates_used"),
        } if att_test_obj is not None else None,
        "es_wcb_p": res_norm.get("es_wcb_p"),
        "wcb_meta": res_norm.get("wcb_meta"),
        "honest_did": {
            "theta_hat": (hd or {}).get("theta_hat", None)
        } if hd else None,
    }
    with open("results/data/summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    def save_covariate_summary_with_vif(
    panel_df: pd.DataFrame,
    covariates: list[str],
    vif_map: dict[str, float],
    path: str,
    caption: str,
    ) -> None:
        """
        Build and save a covariate summary table (mean, std, N, VIF) over pre-treatment observations.

        Parameters
        ----------
        panel_df : DataFrame
            The prepared panel (results.data.panel).
        covariates : list of str
            Names of covariate columns to summarise.
        vif_map : dict
            Mapping from covariate name -> VIF (from preparation.info["covariate_vif"]).
        path : str
            Output .tex path.
        caption : str
            LaTeX table caption.
        """
        # Restrict to pre-treatment observations to match the caption
        if "post" in panel_df.columns:
            df_pre = panel_df[panel_df["post"] == 0].copy()
        else:
            df_pre = panel_df.copy()

        rows = []
        for c in covariates:
            if c not in df_pre.columns:
                continue
            s = df_pre[c]
            rows.append(
                {
                    "Covariate": c,
                    "Mean": s.mean(),
                    "Std": s.std(),
                    "N": s.notna().sum(),
                    "VIF": vif_map.get(c, np.nan),
                }
            )
        cov_df = pd.DataFrame(rows)

        # Write LaTeX with all columns, including VIF
        tex = cov_df.to_latex(
            index=False,
            escape=False,
            caption=caption,
            label=None,
            float_format="%.3f",
        )
        Path(path).write_text(tex, encoding="utf-8")


    # ----------------------------
    # LaTeX tables
    # ----------------------------
    def _write_tex(df: pd.DataFrame, path: str, caption: str) -> None:
        tex = df.to_latex(index=False, escape=False, caption=caption, label=None)
        Path(path).write_text(tex, encoding="utf-8")

    # ATT summary table: manuscript-facing output uses the Callaway-Sant'Anna
    # estimator implemented through the `differences` package.
    att_tbl = pd.DataFrame()
    if summary.get("att_test"):
        att_tbl = pd.DataFrame(
            [
                {
                    "Parameter": r"ATT$^{\circ}$ (Callaway--Sant'Anna)",
                    "Coef": summary["att_test"]["att_overall"],
                    "SE": summary["att_test"]["se_overall"],
                    "p": summary["att_test"]["p_overall"],
                    "p (boot)": summary["att_test"]["p_boot"],
                    "MDE": summary["att_test"]["mde"],
                    "Clusters": summary["att_test"]["clusters"],
                    "n": summary["att_test"]["n"],
                }
            ]
        )
    _write_tex(att_tbl, "results/tex/tab_att_summary.tex", "ATT summary")

    # Event-study coefficients table
    if "df_es" in locals():
        es_tbl = df_es[["event_time", "beta", "lo", "hi", "se", "p"]]
        _write_tex(es_tbl, "results/tex/tab_event_study_coefs.tex", "Event-study coefficients")

    # Support counts table
    if panel_df is not None and {"event_time", "unit_id"}.issubset(panel_df.columns):
        support_tbl = (
            panel_df.dropna(subset=["event_time"])
            .groupby("event_time")["unit_id"]
            .nunique()
            .reset_index(name="units")
        )
        _write_tex(support_tbl, "results/tex/tab_support_counts.tex", "Support by event time")

    # Covariate summary (optional, mean and sd)
        # Covariate summary (pre-treatment means, std, N, and VIFs)
    data_obj = getattr(results, "data", None)
    covs = getattr(data_obj, "covar_cols_used", []) if data_obj is not None else []
    info = getattr(data_obj, "info", {}) if data_obj is not None else {}
    vif_map = info.get("covariate_vif", {}) if isinstance(info, dict) else {}

    if panel_df is not None and covs:
        save_covariate_summary_with_vif(
            panel_df=panel_df,
            covariates=covs,
            vif_map=vif_map,
            path="results/tex/tab_covariate_summary.tex",
            caption=(
                "Summary statistics for covariates over pre-treatment observations. "
                "Means and standard deviations are computed on the standardised covariates; "
                "VIF denotes the variance inflation factor computed on the same sample."
            ),
        )


    print("Done. Outputs are under results/.")


if __name__ == "__main__":
    main()
