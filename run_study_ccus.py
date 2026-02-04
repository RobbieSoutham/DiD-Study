"""
Run the CCUS DiD study end-to-end using the checked-in pipeline.

Usage (ensuring the new_venv environment is used):
    new_venv\\Scripts\\python run_study_ccus.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import seaborn as sns

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

    # ATT pooled
    if hasattr(res, "att"):
        out["att"] = res.att
    elif hasattr(res, "att_pooled"):
        out["att"] = res.att_pooled  # dict-like

    # ATT differences
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

    covariates = [
        "Demand_heat",
        "Demand_renewables_and_waste",
        "energy_demand_fossil_fuels",
        "CPI_growth",
        "GDP_per_capita_PPP",
    ]
    
    cfg = StudyConfig(
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
        use_wcb=True,
        min_cluster_support=2,
        honestdid_enable=False,
    )
    sns.set_theme()
    # Try to load cached study; otherwise run and save
    cache_path = Path("lateststudys.pkl")
    results = None
    if cache_path.exists():
        try:
            results = pickle.load(cache_path.open("rb"))
            print(f"[2/4] Loaded cached study from {cache_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[2/4] Failed to load cache ({exc}); running study ...")

    if results is None:
        print("[2/4] Running DidStudy ... (this can take a while)")
        study = DidStudy(cfg)
        results = study.run()
        print("       Study run complete.")
        try:
            pickle.dump(results, cache_path.open("wb"))
            print(f"       Saved study to {cache_path}")
        except Exception as exc:  # noqa: BLE001
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
    
    # Dose distribution histogram (treated observations)
    dose_series = None
    if panel_df is not None and "dose_level" in panel_df.columns:
        dose_series = panel_df.loc[panel_df["treated_now"] == 1, "dose_level"].astype(float)
    elif isinstance(getattr(getattr(results, "data", None), "info", None), dict):
        dose_series = getattr(results.data, "info", {}).get("dose_series", None)
    if dose_series is not None and len(dose_series) > 0:
        # Standard histogram (no manual dose bins)
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
    att_obj = res_norm.get("att")
    summary = {
        "att": {
            "coef": _get(att_obj, "coef"),
            "se": _get(att_obj, "se"),
            "p": _get(att_obj, "p"),
            "p_wcb": _get(att_obj, "p_wcb"),
            "clusters": _get(att_obj, "clusters"),
            "n": _get(att_obj, "n"),
            "mde": _get(att_obj, "mde"),
        },
        "att_test": {
            "att_overall": _get(att_test_obj, "att_overall"),
            "se_overall": _get(att_test_obj, "se_overall"),
            "p_overall": _get(att_test_obj, "p_overall"),
            "p_boot": _get(att_test_obj, "p_boot"),
            "p_wcb": _get(att_test_obj, "p_wcb"),
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

    # ATT summary table
    att_tbl = pd.DataFrame(
        [
            {
                "Parameter": "ATT$^{o}$ (long-diff)",
                "Coef": summary["att"]["coef"],
                "SE": summary["att"]["se"],
                "p": summary["att"]["p"],
                "p (WCB)": summary["att"]["p_wcb"],
                "Clusters": summary["att"]["clusters"],
                "n": summary["att"]["n"],
                "MDE": summary["att"]["mde"],
            }
        ]
    )
    if summary.get("att_test"):
        att_tbl = pd.concat(
            [
                att_tbl,
                pd.DataFrame(
                    [
                        {
                            "Parameter": "ATT$^{o}$ (differences)",
                            "Coef": summary["att_test"]["att_overall"],
                            "SE": summary["att_test"]["se_overall"],
                            "p": summary["att_test"]["p_overall"],
                            "p (WCB/boot)": summary["att_test"]["p_wcb"],
                            "Clusters": "",
                            "n": "",
                            "MDE": "",
                        }
                    ]
                ),
            ],
            ignore_index=True,
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
