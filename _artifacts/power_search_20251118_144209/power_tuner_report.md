# CCUS-informed Power Tuner (2–3 bins, VIF)
Target MDE: 0.015; alpha=0.05
MDE formula: (t_{1−α/2, G−1} + t_{power, G−1}) × SE

## Executive Summary
No — no configuration reaches MDE < |effect|.
Counts — meeting: 0, highlighted (significant): 0, PTA+WCB clean: 0, HonestDiD-pass: 0.

## Top Configurations
- coef=0.0114, se=0.01285, mde=0.03613, mde_ratio=0.3156; p=0.37496637610655337, p_wcb=nan, pta_p=0.18456267117692093, wcb_pre=nan; n_clusters=298, post_share=0.012055455093429777; dose_quantiles=[0.0, 0.5, 1.0], n_bins=2 

## Drivers (summary)
- Intercept: coef=4.067e-07, p=1
- C(use_log_outcome)[T.True]: coef=4.067e-07, p=1
- C(differenced)[T.True]: coef=4.067e-07, p=1
- C(use_lag_levels_in_diff)[T.True]: coef=4.067e-07, p=1
- min_pre: coef=8.135e-07, p=1