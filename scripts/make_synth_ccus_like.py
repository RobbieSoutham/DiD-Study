from __future__ import annotations

import numpy as np
import pandas as pd


def make_synth_ccus_like(
    *,
    n_countries: int = 80,
    n_sectors: int = 6,
    years=range(2000, 2004),
    seed: int = 1234,
) -> pd.DataFrame:
    """Create a CCUS-like synthetic panel compatible with PanelData.

    Key requirements for our pipeline:
      - columns: Year, Country, Sector, CCUS_sector, emissions_sector, Emissions, eor_capacity
      - plus covariates used in run_study_ccus.py
      - balanced panel

    We generate staggered adoption + time-invariant dose per unit (required by contdid).
    We still store eor_capacity as constant within unit across all periods
    (including pre), matching contdid convention.
    """

    rng = np.random.default_rng(seed)
    years = list(years)

    countries = [f"C{i:03d}" for i in range(1, n_countries + 1)]
    sectors = [f"S{j:02d}" for j in range(1, n_sectors + 1)]

    units = [(c, s) for c in countries for s in sectors]

    # adoption timing group (first treated year) or never treated
    # roughly 15% treated
    treat_flag = rng.random(len(units)) < 0.15
    g_choices = rng.choice(years[1:], size=len(units))  # avoid first year
    g = np.where(treat_flag, g_choices, 0)

    # time-invariant dose for treated units
    # mixture to create rich support
    dose = np.where(
        treat_flag,
        rng.lognormal(mean=0.0, sigma=0.7, size=len(units)),
        0.0,
    )

    # baseline unit effect + common time shocks
    alpha = rng.normal(0.0, 0.4, size=len(units))
    time_shock = {yr: rng.normal(0.0, 0.15) for yr in years}

    rows = []
    for idx, (country, ccus_sector) in enumerate(units):
        for yr in years:
            # event time
            tau = (yr - g[idx]) if g[idx] != 0 else None

            # nonlinear dose-response after adoption
            post = (g[idx] != 0) and (yr >= g[idx])
            d = float(dose[idx])
            # true effect: concave nonlinear in dose
            te = 0.0
            if post:
                te = (-0.25 * d) + (-0.15 * (d**2))

            eps = rng.normal(0.0, 0.5)
            y_level = 5.0 + alpha[idx] + time_shock[yr] + te + eps
            emissions = float(np.exp(y_level))

            # For the *pipeline* we must encode capacity as 0 pre-adoption and >0 post.
            # (Panel preparation uses this to construct treated_now, event_time, etc.)
            cap_it = float(d) if post else 0.0

            rows.append(
                {
                    "Year": yr,
                    "Country": country,
                    "Sector": ccus_sector,
                    "CCUS_sector": ccus_sector,
                    "emissions_sector": ccus_sector,
                    "Emissions": emissions,
                    "eor_capacity": cap_it,
                    # covariates (dummy but non-constant)
                    "Demand_heat": float(rng.normal(0.0, 1.0)),
                    "Demand_renewables_and_waste": float(rng.normal(0.0, 1.0)),
                    "energy_demand_fossil_fuels": float(rng.normal(0.0, 1.0)),
                    "CPI_growth": float(rng.normal(0.02, 0.01)),
                    "GDP_per_capita_PPP": float(rng.normal(30000, 5000)),
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_synth_ccus_like()
    out = "results/synth_ccus_like.csv"
    df.to_csv(out, index=False)
    print(f"OK: wrote {len(df):,} rows -> {out}")
