"""
Test script for the ``did_study`` package.

This script constructs a simple synthetic data set, configures and runs
the difference‑in‑differences analysis using :class:`did_study.study.DidStudy`,
and prints a concise summary of the results.  It serves as a smoke test
for the full pipeline and illustrates how to use the configuration
options described in the documentation.

The synthetic data set contains two countries (A and B), two CCUS
sectors (Power and Cement) and six calendar years.  One cluster
(Country A, Power) adopts treatment at year 2013 with a dose level
increasing thereafter; all other clusters remain untreated.  Covariate
values are generated deterministically for reproducibility.

Usage
-----
Run this script with Python from the project root::

    python test_run.py

The output will display the panel statistics, pooled and binned ATT
estimates and event study coefficients.  It relies only on
dependencies declared in ``setup.py`` and does not require ``rpy2``.

"""

import pandas as pd
from did_study.helpers.config import StudyConfig
from did_study.study import DidStudy
from did_study.reporting import print_study_summary


def build_synthetic_data() -> pd.DataFrame:
    """Construct a small synthetic data set for demonstration.

    Returns
    -------
    pandas.DataFrame
        A panel with columns required by :func:`prepare_ccus_panel`.
    """
    # Define basic structure
    countries = ["A", "A", "B", "B"]
    sectors = ["Power", "Cement", "Power", "Cement"]
    years = list(range(2010, 2016))  # 6 years
    data = []
    for country, sector in zip(countries, sectors):
        for year in years:
            # baseline emissions
            base_emis = 100.0 if sector == "Power" else 60.0
            # dose/adoption: treat A–Power after 2012
            dose = 0.0
            if country == "A" and sector == "Power" and year >= 2013:
                dose = 1.0 * (year - 2012)
            # emissions grow slightly with year and treatment
            emis = base_emis + 2.0 * (year - 2010) + 5.0 * dose
            data.append(
                {
                    "Country": country,
                    "CCUS_sector": sector,
                    "Sector": "Energy" if sector == "Power" else "Industry",
                    "emissions_sector": "Main Activity Electricity and Heat Production"
                    if sector == "Power"
                    else "Cement production",
                    "Year": year,
                    "Emissions": emis,
                    "eor_capacity": dose,
                    # covariates (simple deterministic functions)
                    "Demand_electricity": 50 + (year - 2010) * 0.5,
                    "Demand_heat": 20 + (year - 2010) * 0.2,
                    "Demand_nuclear": 10 + (year - 2010) * 0.1,
                    "Demand_renewables_and_waste": 5 + (year - 2010) * 0.05,
                    "Supply_nuclear": 15 + (year - 2010) * 0.1,
                    "energy_demand_fossil_fuels": 30 + (year - 2010) * 0.3,
                    "CPI_growth": 2.0 + 0.1 * (year - 2010),
                    "GDP_per_capita_PPP": 40000 + 500 * (year - 2010),
                    "renewable_to_fossil_supply_ratio": 0.2 + 0.01 * (year - 2010),
                }
            )
    return pd.DataFrame(data)


def main() -> None:
    # build synthetic data
    df = build_synthetic_data()
    # configuration mirroring the example provided in the question
    covariates = [
        "Demand_electricity",
        "Demand_heat",
        "Demand_nuclear",
        "Demand_renewables_and_waste",
        "Supply_nuclear",
        "energy_demand_fossil_fuels",
        "CPI_growth",
        "GDP_per_capita_PPP",
        "renewable_to_fossil_supply_ratio",
    ]
    cfg = StudyConfig(
        df=df,
        outcome_mode="total",  # collapse emissions to country level
        supdem_mode="sum",
        mapping=None,
        unit_cols=("Country", "CCUS_sector"),
        year_col="Year",
        outcome_col="Emissions",
        emissions_sector_col="emissions_sector",
        capacity_col="eor_capacity",
        sector_col="Sector",
        covariates=covariates,
        differenced=True,
        use_log_outcome=True,
        min_pre=2,
        min_post=1,
        absorb_treatment=True,
        n_bins=3,
        treat_threshold=0.0,
        binning_mode="quantile",
        pre=3,
        post=3,
        use_wcb=False,  # disable bootstrap for speed in test
        wcb_B=999,
        wcb_weights="auto",
        min_cluster_support=1,
        optimise_ds=False,
    )
    study = DidStudy(cfg)
    results = study.run()
    print_study_summary(results)


if __name__ == "__main__":
    main()