from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from did_study.helpers.config import StudyConfig
from did_study.study import DidStudy
from did_study.reporting.plotting import (
    plot_event_study_line,
    plot_contdid_att_gt,
    plot_contdid_dose_response,
    plot_dose_distribution,
)


def main() -> None:
    Path("results/synth_contdid").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("results/synth_ccus_like.csv")

    covariates = [
        "Demand_heat",
        "Demand_renewables_and_waste",
        "energy_demand_fossil_fuels",
        "CPI_growth",
        "GDP_per_capita_PPP",
    ]

    # Note: outcome_mode can be a column name (custom outcome) to avoid mapping.
    cfg = StudyConfig(
        df=df,
        outcome_mode="Emissions",
        supdem_mode="direct",
        mapping=None,
        unit_cols=("Country", "CCUS_sector"),
        year_col="Year",
        outcome_col="Emissions",
        emissions_sector_col="emissions_sector",
        capacity_col="eor_capacity",
        sector_col="Sector",
        covariates=covariates,
        differenced=False,
        use_log_outcome=True,
        min_pre=1,
        min_post=1,
        n_bins=3,
        dose_bins=[0, 0.5, 1.5, np.inf],
        treat_threshold=0,
        pre=2,
        post=2,
        cluster_col="unit_id",
        use_wcb=False,
        honestdid_enable=False,
        contdid_enable=True,
        artifact_dir="results/synth_contdid",
    )

    study = DidStudy(cfg)
    res = study.run()

    # contdid artifacts are written by the R runner to artifact_dir/contdid
    contdid_dir = Path(cfg.artifact_dir) / "contdid"

    # Plot dose distribution (treated rows)
    panel_df = res.data.panel
    dose_series = panel_df.loc[panel_df["treated_now"] == 1, "dose_level"].astype(float)
    plot_dose_distribution(
        dose_series,
        title=None,
        xlabel="Dose (synthetic)",
        ylabel="Units",
        save=str(Path(cfg.artifact_dir) / "dose_distribution.png"),
        show=False,
    )

    # contdid eventstudy + ATT(g,t)
    es_path = contdid_dir / "contdid_eventstudy.csv"
    gt_path = contdid_dir / "contdid_att_gt.csv"

    if es_path.exists():
        df_es = pd.read_csv(es_path)
        # Rename contdid columns to match plot_event_study_line convention
        df_es = df_es.rename(columns={"tau": "event_time", "att": "beta"})
        if "se" in df_es.columns:
            df_es["lo"] = df_es["beta"] - 1.96 * df_es["se"]
            df_es["hi"] = df_es["beta"] + 1.96 * df_es["se"]
        plot_event_study_line(
            df_es,
            title=None,
            xlabel=r"Event time ($\tau$)",
            ylabel="ACRT(τ)",
            ref_tau=-1,
            save=str(Path(cfg.artifact_dir) / "contdid_eventstudy.png"),
            show=False,
        )

    if gt_path.exists():
        df_gt = pd.read_csv(gt_path)
        plot_contdid_att_gt(
            df_gt,
            title=None,
            xlabel=r"Event time ($\tau$)",
            ylabel="ATT(g,t)",
            save=str(Path(cfg.artifact_dir) / "contdid_att_gt.png"),
            show=False,
        )

    dose_path = contdid_dir / "contdid_dose_response.csv"
    if dose_path.exists():
        df_dose = pd.read_csv(dose_path)
        if len(df_dose) > 0:
            plot_contdid_dose_response(
                df_dose,
                title=None,
                xlabel="Dose (Mt CO$_2$/yr)",
                ylabel="ATT(d) / ACRT",
                save=str(Path(cfg.artifact_dir) / "contdid_dose_response.png"),
                show=False,
            )

    # Copy summary json into synth_contdid root for convenience
    summ = contdid_dir / "summary.json"
    if summ.exists():
        (Path(cfg.artifact_dir) / "contdid_summary.json").write_text(summ.read_text(), encoding="utf-8")

    print("OK: wrote synth contdid figures to", cfg.artifact_dir)


if __name__ == "__main__":
    main()
