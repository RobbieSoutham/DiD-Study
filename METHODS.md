# Methods and Identification

This document formalizes the analysis pipeline implemented in the `did_study` package (see `did_study/study.py` and associated estimators). It revises earlier drafts that relied on ATT^o aggregation from Callaway–Sant’Anna and TWFE without fixed effects, and aligns the write‑up with the code that is now used in all results and figures.

## Data and Integration

- Sources: IEA CCUS Projects Database (capture capacity), IEA World Energy Balances (supply/demand), EDGAR v8.0 (CO2 emissions), and IMF WEO (macro covariates) \cite{ieaCCUS2024,ieaWorldEnergyBalances2024,edgar2024,imfWEO2024}.
- Units and years: Country×CCUS sector ("direct" outcome mode) or country totals ("total" mode); annual 1998–2022.
- Emissions allocation (direct mode): Territorial emissions by emissions sector are mechanically allocated to CCUS sectors via a partition (or weighted mapping) and aggregated to unit level. This preserves consistency with territorial accounting while aligning outcomes to the capture locus.
- Energy aggregates: Sectoral `Supply_*/Demand_*` series are pivoted to country–year wide form (with totals), enabling composites and principal‑components indexing of fossil demand.

## Panel Construction, Treatment, Event Time, and Support

- Dose and adoption: For unit i, year t, define a non‑negative dose `dose_level_it` from summed CCUS capacity. With threshold τ≥0 (τ=0 in main runs), adoption is `adopt_now_it = 1{ dose_level_it > τ }`. The first adoption year is `g_i = min{ t : adopt_now_it = 1 }`. Post periods are `post_it = 1{ t ≥ g_i }`. Never‑treated units have `g_i = NA` and `post_it = 0`.
- Event time: For ever‑treated units, `event_time_it = t − g_i` and missing otherwise.
- Dose bins (heterogeneity): To summarize dose heterogeneity, we compute bins on the positive part of the post‑treatment dose distribution using either explicit edges, explicit quantiles, or equal‑frequency quantiles when `n_bins` is set. Each treated unit is assigned to its first‑post bin (absorbing cohort), and `dose_bin` is populated only on post rows (pre rows keep NA). Support (units/rows per bin) is reported at post.
- Trimming for support: Ever‑treated units require at least `min_pre` pre‑treatment and `min_post` post periods; never‑treated units require at least `min_pre + min_post` distinct years. Event‑time horizons are further filtered to ensure at least `min_cluster_support` treated clusters at each τ.

## Outcome and Covariates

- Outcome transform: Let `y_it` be unit emissions. We use `log(y_it + ε)` with small ε>0; differencing the log is optional. In the main runs, we use log levels (not first differences), which triggers unit and year fixed effects by default.
- Covariates: A small, interpretable set of time‑varying controls:
  - Energy mix: `renewable_to_fossil_supply_ratio` and `nuclear_share_supply` (bounded to [0,1]); we also expose its lag and difference for diagnostics.
  - Fossil demand index (PC1) from four fossil demand blocks, fit on pre rows only to avoid leakage \cite{jolliffe2016}.
  - Requested macro covariates are resolved from source columns (wide blocks, totals, or raw series), transformed as levels/lagged/differenced according to the outcome transform, and standardized.
  - Scaling and fitting are performed on pre rows only (never‑ or not‑yet‑treated) and then applied to the full panel (leakage‑safe). We report VIFs as a descriptive diagnostic.
- Bad‑control caution: Because some covariates may respond to treatment, we use pre‑fit scaling, expose lag/diff variants, and treat event‑study as a diagnostic; HonestDiD bounds guard against modest deviations from parallel trends \cite{rambachanHonestDiD2023}.

## Estimands and Estimators

### Identification

The core identifying condition is generalized parallel trends conditional on unit and year fixed effects and observed covariates. Under heterogeneity and staggered adoption, pooled two‑way fixed effects (TWFE) average many cohort×time effects; weights can be non‑convex \cite{chaisemartinTwoWayFixed2020,goodmanDifferenceinDifferencesEstimation2021,SunAbraham2021,callawayDifferenceinDifferencesEstimation2021}. We therefore present pooled estimates as descriptive summaries and emphasize event‑study diagnostics and sensitivity analyses.

### Pooled ("ATT^o", TWFE coefficient)

Main specification:

`log(y_it) = α + θ · treated_now_it + Γ′ X_it + μ_i + λ_t + ε_it,`

with unit (μ_i) and year (λ_t) fixed effects; `X_it` are the scaled covariates above. If the outcome is a first difference (names beginning `d_`), unit fixed effects are omitted by default; for log levels they are included. We label θ in tables as “ATT^o (pooled)” for continuity, but it is the TWFE coefficient rather than the Callaway–Sant’Anna ATT^o estimand; with constant effects it coincides with ATT.

### Dose‑bin heterogeneity (absorbing cohorts)

We estimate bin‑specific effects by replacing the single treatment indicator with bin dummies (reference = “untreated”):

`log(y_it) = α + Σ_b θ_b · 1{ unit_i in bin b & post_it = 1 } + Γ′ X_it + μ_i + λ_t + ε_it.`

We report analytic and wild‑cluster p‑values, cluster counts, and MDEs per bin.

### Event‑study (diagnostic dynamics)

We estimate an event‑time regression over a window [−pre,…,−2,0,1,…,post] excluding −1 as reference:

`log(y_it) = α + Σ_{τ≠−1} β_τ · 1{ event_time_it = τ & ever treated } + Γ′ X_it + μ_i + λ_t + ε_it,`

including the same covariates and fixed effects as the pooled model. Following \cite{SunAbraham2021}, we treat ES as descriptive diagnostics under staggered timing and heterogeneity rather than as an identifying estimator. We report β_τ, a joint pre‑trends F‑test on leads τ≤−2, and a joint wild‑cluster test across all ES coefficients.

## Inference, Bootstrap, and Power

- Analytic SEs are clustered at the unit level (the serially correlated panel dimension), consistent with best practice in DID panels with many periods \cite{Bertrand2004}.
- Wild cluster bootstrap (WCB) p‑values are obtained using `fwildclusterboot` on a `fixest::feols` fit. We choose Rademacher weights except for 5–12 clusters, where Webb’s six‑point weights are used by default \cite{cameronBootstrapBasedImprovements2008,mackinnonWildBootstrapInference2022}. Replications `B` follow the weight choice (large by default) unless overridden in configuration. Scalar tests use `boottest`; joint tests (bins or all ES coefficients) use `mboottest`.
- Minimum detectable effects (MDEs) are computed from the cluster‑robust SE and a t‑distribution with df = G−1 (G = number of clusters) to contextualize power \cite{cameronBootstrapBasedImprovements2008}.

## Sensitivity to Parallel‑Trends Violations (HonestDiD)

We use HonestDiD relative‑magnitude (Δ^RM) sensitivity bounds \cite{rambachanHonestDiD2023}. Let β_pre and β_post be ES coefficients ordered by event time; define θ = `l′ β_post`, where `l` are exposure weights proportional to the number of treated observations at each post‑event horizon in the estimation window. Given the ES covariance matrix, we compute bounds `[θ_L(M), θ_U(M)]` on a grid `M ∈ [0, M_max]`, where `M` parameterizes how large post‑treatment violations are allowed to be relative to pre‑treatment deviations. We report the naive `θ̂` (plug‑in using `l`) and bounds across the M‑grid.

## Assumptions, Limitations, External Validity

- Parallel trends: Identification relies on generalized parallel trends conditional on fixed effects and observed covariates. Pre‑trend tests are low‑power diagnostics rather than decision rules \cite{rothPretestCaution2022}.
- Heterogeneous effects and staggered timing: TWFE pooled estimates average heterogeneous cohort×time effects and can carry non‑convex weights; we therefore treat the pooled coefficient as descriptive and emphasize event‑study diagnostics, bin heterogeneity, MDEs, and HonestDiD \cite{chaisemartinTwoWayFixed2020,goodmanDifferenceinDifferencesEstimation2021,SunAbraham2021,callawayDifferenceinDifferencesEstimation2021}.
- Continuous treatment: Our heterogeneity analysis uses absorbing dose bins. Formal identification of dose–response in staggered DiD with continuous treatments calls for specialized estimators; see \cite{callawayDifferenceindifferencesContinuousTreatment2024}. We defer dose–response estimation to future work.
- External validity: Effects are identified for the study sample and support; transport to other institutional or energy‑mix contexts should be cautious \cite{findleyExternalValidity2021}.

## Reporting Conventions

- Pooled and bin‑specific outputs: coefficients, analytic SEs with cluster counts, analytic and WCB p‑values (test type and weights), and MDEs.
- Event study: β_τ with SEs and 95% CIs, a joint analytic F‑test of pre‑trends (τ≤−2), and a joint WCB test across all ES coefficients; support by τ and by bin.
- HonestDiD: M‑grid, lower/upper bounds, exposure weights `l`, naive `θ̂`.

## Alignment with the Literature

The implemented pipeline reflects current best practice for staggered‑adoption DID with few treated clusters and potential heterogeneity: (i) TWFE is accompanied by explicit caveats and is not over‑interpreted \cite{chaisemartinTwoWayFixed2020,goodmanDifferenceinDifferencesEstimation2021,SunAbraham2021}; (ii) diagnostics use event studies without elevating pretests to decision rules \cite{rothPretestCaution2022}; (iii) inference relies on cluster‑robust SEs and wild‑cluster bootstrap suited to few clusters \cite{Bertrand2004,cameronBootstrapBasedImprovements2008,mackinnonWildBootstrapInference2022}; and (iv) HonestDiD provides transparent sensitivity to mild violations of parallel trends \cite{rambachanHonestDiD2023}. The choice to summarize continuous exposure via absorbing dose bins is explicitly framed as a heterogeneity analysis rather than a structural dose–response estimator, consistent with recent guidance \cite{callawayDifferenceindifferencesContinuousTreatment2024}. Finally, pre‑fit scaling and construction of covariates on pre rows only mitigate leakage.

## References

See `citations.bib` for full entries: \cite{ieaCCUS2024,ieaWorldEnergyBalances2024,edgar2024,imfWEO2024,callawayDifferenceinDifferencesEstimation2021,SunAbraham2021,goodmanDifferenceinDifferencesEstimation2021,chaisemartinTwoWayFixed2020,rothPretestCaution2022,rambachanHonestDiD2023,cameronBootstrapBasedImprovements2008,mackinnonWildBootstrapInference2022,jolliffe2016,findleyExternalValidity2021,callawayDifferenceindifferencesContinuousTreatment2024}.

