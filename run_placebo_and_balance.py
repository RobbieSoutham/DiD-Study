"""
R1.1 additions: extended placebo timing test (1000 draws) + covariate balance
table (treated vs untreated, pre-treatment). Reuses the existing pipeline
helpers; skips WCB, HonestDiD, LOO to stay quick. Outputs:

    results/data/placebo_twfe_distribution.csv      (overwrites)
    results/data/placebo_twfe_summary.json
    results/figures/placebo_twfe_distribution.png   (overwrites)
    results/data/covariate_balance_treated_vs_untreated.csv
    results/tex/tab_covariate_balance.tex
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Suppress expensive steps before importing the pipeline modules.
os.environ.setdefault("CCUS_SKIP_LOO", "1")
os.environ.setdefault("CCUS_SKIP_WCB", "1")
os.environ.setdefault("CCUS_SKIP_HONESTDID", "1")

import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make the run_study_ccus helpers importable.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_study_ccus import _build_config, _within_twfe_coef  # noqa: E402
from did_study.study import DidStudy  # noqa: E402
from did_study.reporting.style import (  # noqa: E402
    COL_POOLED,
    apply_paper_theme,
    zero_line,
)

PAPER_FIG_DIR = (
    HERE.parent.parent / "papers" / "b_ccus_did" / "figures"
).resolve()

RNG_SEED = 123
PLACEBO_DRAWS = int(os.environ.get("CCUS_PLACEBO_DRAWS", "1000"))


def main() -> None:
    sns.set_theme()
    Path("results/data").mkdir(parents=True, exist_ok=True)
    Path("results/tex").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading raw panel ds_no_prep.csv ...")
    raw_df = pd.read_csv("ds_no_prep.csv")
    print(f"       rows: {len(raw_df):,}")

    print(f"[2/4] Preparing panel (skipping WCB/HonestDiD/LOO) ...")
    cfg = _build_config(raw_df, honestdid_enable=False)
    study = DidStudy(cfg)
    result = study.run(run_event_study=False, run_honest_did=False)
    panel_df = getattr(getattr(result, "data", None), "panel", None)
    if panel_df is None:
        panel_df = getattr(result, "panel_df", None)
    if panel_df is None:
        raise RuntimeError("Could not locate prepared panel on DidStudy result.")
    print(f"       prepared panel rows: {len(panel_df):,}; "
          f"unique units: {panel_df['unit_id'].nunique()}")

    # ------------------------------------------------------------------
    # Placebo timing test
    # ------------------------------------------------------------------
    print(f"[3/4] Placebo timing test with {PLACEBO_DRAWS} random draws ...")
    rng = np.random.default_rng(RNG_SEED)
    treated_keys = (
        panel_df.loc[panel_df["treated_ever"] == 1, ["unit_id", "g"]]
        .drop_duplicates()
        .dropna()
    )
    n_treated = len(treated_keys)
    adoption_years = treated_keys["g"].astype(int).to_numpy()
    control_units = sorted(
        panel_df.loc[panel_df["treated_ever"] == 0, "unit_id"].dropna().unique()
    )
    if n_treated == 0 or len(control_units) < n_treated:
        raise RuntimeError("Not enough control units to draw a placebo sample.")
    print(f"       treated units: {n_treated}; control pool: {len(control_units)}; "
          f"adoption-year support: {sorted(set(adoption_years))}")

    # Observed TWFE coefficient on the real binary treatment indicator.
    observed = _within_twfe_coef(
        panel_df.assign(placebo_treat=panel_df["post"]),
        treatment_col="placebo_treat",
    )
    print(f"       observed TWFE coefficient: {observed:.4f}")

    rows = [{"draw": -1, "att_twfe": observed, "kind": "observed"}]
    for i in range(PLACEBO_DRAWS):
        selected = rng.choice(control_units, size=n_treated, replace=False)
        assigned_years = rng.choice(adoption_years, size=n_treated, replace=True)
        g_map = dict(zip(selected, assigned_years))
        placebo_g = panel_df["unit_id"].map(g_map)
        d = (
            (placebo_g.notna())
            & (panel_df["Year"].astype(int) >= placebo_g.astype("float"))
        ).astype(int)
        coef = _within_twfe_coef(
            panel_df.assign(placebo_treat=d),
            treatment_col="placebo_treat",
        )
        rows.append({"draw": i, "att_twfe": coef, "kind": "placebo"})

    out = pd.DataFrame(rows)
    placebo_vals = out.loc[out["kind"] == "placebo", "att_twfe"].dropna()
    upper_p = float((placebo_vals >= observed).mean())
    two_sided_p = float((placebo_vals.abs() >= abs(observed)).mean())
    out["observed_upper_tail_share"] = upper_p
    out["observed_two_sided_p"] = two_sided_p
    out.to_csv("results/data/placebo_twfe_distribution.csv", index=False)

    summary = {
        "observed_twfe": float(observed),
        "n_placebo_draws": int(len(placebo_vals)),
        "placebo_mean": float(placebo_vals.mean()),
        "placebo_sd": float(placebo_vals.std()),
        "placebo_q025": float(placebo_vals.quantile(0.025)),
        "placebo_median": float(placebo_vals.median()),
        "placebo_q975": float(placebo_vals.quantile(0.975)),
        "one_sided_upper_p": upper_p,
        "two_sided_empirical_p": two_sided_p,
        "n_treated": int(n_treated),
        "n_controls": int(len(control_units)),
        "seed": RNG_SEED,
    }
    with open("results/data/placebo_twfe_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"       placebo mean = {summary['placebo_mean']:.4f}, "
          f"sd = {summary['placebo_sd']:.4f}, "
          f"one-sided upper p = {upper_p:.4f}, two-sided p = {two_sided_p:.4f}")

    apply_paper_theme()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(placebo_vals, bins=40, color="0.78", edgecolor="white")
    zero_line(ax, orient="v")
    ax.axvline(
        observed,
        color=COL_POOLED,
        linewidth=2,
        label=(
            r"Observed TWFE = "
            f"{observed:.3f}  "
            f"(one-sided $p$ = {upper_p:.3f}; two-sided $p$ = {two_sided_p:.3f})"
        ),
    )
    ax.set_xlabel("Placebo within-transform coefficient")
    ax.set_ylabel("Number of draws")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig_local = "results/figures/placebo_twfe_distribution.png"
    fig.savefig(fig_local, dpi=220)
    plt.close(fig)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fig_local, PAPER_FIG_DIR / "placebo_twfe_distribution.png")
    print(f"       wrote {fig_local} and copied to paper figures/")

    # ------------------------------------------------------------------
    # Covariate balance: treated vs untreated, pre-treatment only
    # ------------------------------------------------------------------
    print(f"[4/4] Building covariate balance table (pre-treatment) ...")
    covariates = list(cfg.covariates or [])
    pre = panel_df[panel_df["post"] == 0].copy() if "post" in panel_df.columns \
        else panel_df.copy()
    treated_mask = pre["treated_ever"] == 1
    untreated_mask = pre["treated_ever"] == 0

    bal_rows = []
    for c in covariates:
        if c not in pre.columns:
            continue
        t = pre.loc[treated_mask, c].astype(float)
        u = pre.loc[untreated_mask, c].astype(float)
        t_mean, t_sd, t_n = t.mean(), t.std(), int(t.notna().sum())
        u_mean, u_sd, u_n = u.mean(), u.std(), int(u.notna().sum())
        # Standardised mean difference (Cohen's d using pooled SD).
        pooled_sd = np.sqrt(((t_n - 1) * t_sd**2 + (u_n - 1) * u_sd**2) / max(t_n + u_n - 2, 1))
        smd = (t_mean - u_mean) / pooled_sd if pooled_sd > 0 else np.nan
        bal_rows.append({
            "covariate": c,
            "treated_mean": t_mean,
            "treated_sd": t_sd,
            "treated_n": t_n,
            "untreated_mean": u_mean,
            "untreated_sd": u_sd,
            "untreated_n": u_n,
            "mean_diff": t_mean - u_mean,
            "std_mean_diff": smd,
        })
    bal_df = pd.DataFrame(bal_rows)
    bal_df.to_csv("results/data/covariate_balance_treated_vs_untreated.csv", index=False)
    print(bal_df.to_string(index=False))

    # Friendly labels for the .tex.
    label_map = {
        "Demand_heat": "Heat demand",
        "Demand_renewables_and_waste": "Ren. \\& waste demand",
        "energy_demand_fossil_fuels": "Fossil demand (PC1)",
        "CPI_growth": "CPI inflation",
        "GDP_per_capita_PPP": "GDP per capita (PPP)",
    }

    rows_tex = []
    rows_tex.append(r"\begin{table}[ht!]")
    rows_tex.append(r"\centering")
    rows_tex.append(r"\caption{Pre-treatment covariate balance: treated vs untreated "
                    r"\textit{country--sectors}. Means and standard deviations are "
                    r"computed on standardised pre-treatment observations "
                    r"(post indicator equal to zero). The standardised mean "
                    r"difference (SMD) is the Cohen-d-style ratio of the mean "
                    r"difference to the pooled standard deviation.}")
    rows_tex.append(r"\label{tab:covbalance}")
    rows_tex.append(r"\small")
    rows_tex.append(r"\begin{tabular}{lcccccc}")
    rows_tex.append(r"\toprule")
    rows_tex.append(r"\textbf{Covariate} & "
                    r"\multicolumn{2}{c}{\textbf{Treated}} & "
                    r"\multicolumn{2}{c}{\textbf{Untreated}} & "
                    r"\textbf{Diff} & \textbf{SMD} \\")
    rows_tex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    rows_tex.append(r" & Mean & SD & Mean & SD & & \\")
    rows_tex.append(r"\midrule")
    for r in bal_rows:
        lab = label_map.get(r["covariate"], r["covariate"].replace("_", r"\_"))
        rows_tex.append(
            f"{lab} & {r['treated_mean']:.3f} & {r['treated_sd']:.3f} & "
            f"{r['untreated_mean']:.3f} & {r['untreated_sd']:.3f} & "
            f"{r['mean_diff']:.3f} & {r['std_mean_diff']:.3f} \\\\"
        )
    rows_tex.append(r"\bottomrule")
    rows_tex.append(r"\end{tabular}")
    rows_tex.append(r"\end{table}")
    tex = "\n".join(rows_tex) + "\n"
    Path("results/tex/tab_covariate_balance.tex").write_text(tex, encoding="utf-8")
    print("       wrote results/tex/tab_covariate_balance.tex")

    print("Done.")


if __name__ == "__main__":
    main()
