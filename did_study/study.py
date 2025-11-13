from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .helpers.config import StudyConfig
from .helpers.preparation import PanelData
from .estimator import DidEstimator
from .robustness.honest_did import honest_did_bounds
from .helpers.utils import choose_wcb_weights_and_B


@dataclass
class DidStudyResult:
    """Container for all outputs of a DidStudy run."""

    config: StudyConfig
    data: PanelData

    # Core estimators
    att: Optional[Any] = None
    bins: Optional[Any] = None
    event_study: Optional[Any] = None

    # Robustness / inference
    honest_did: Optional[Dict[str, Any]] = None
    es_wcb_p: Optional[float] = None

    # Metadata / diagnostics
    wcb_meta: Dict[str, Any] = field(default_factory=dict)


def _tau_from_es_name(col: str) -> Optional[int]:
    """Map ES_t* column names back to integer event times."""
    if col.startswith("ES_tm"):
        return -int(col.replace("ES_tm", ""))
    if col.startswith("ES_t"):
        return int(col.replace("ES_t", ""))
    return None


class DidStudy:
    """
    High-level orchestration class for a DiD study.

    This class:
      1. Takes a StudyConfig.
      2. Prepares the panel data (constructs treatment indicators, bins, event-time dummies).
      3. Runs the requested estimators (ATT^o, dose-response bins, event study).
      4. Optionally runs robustness procedures (HonestDiD, WCB).
    """

    def __init__(self, config: StudyConfig) -> None:
        self.config = config
        self._estimator: Optional[DidEstimator] = None
        self._panel: Optional[PanelData] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def estimator(self) -> DidEstimator:
        if self._estimator is None:
            raise RuntimeError("Estimator not initialised yet. Call .run().")
        return self._estimator

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        run_att: bool = True,
        run_bins: bool = True,
        run_event_study: bool = True,
        run_honest_did: bool = True,
    ) -> DidStudyResult:
        """
        Run the full DiD study pipeline.
        """
        # 1) Prepare panel
        panel = PanelData(self.config)
        self._panel = panel

        df = panel.panel
        if df is not None and "dose_level" in df.columns:
            info = getattr(panel, "info", {})
            if isinstance(info, dict) and "dose_series" not in info:
                info["dose_series"] = df["dose_level"].astype(float).dropna()

        # 2) Choose WCB config once
        G_total = int(df["unit_id"].nunique())
        G_treated = int(df.loc[df["treated_ever"] == 1, "unit_id"].nunique())
        wcb_weights_auto, wcb_B = choose_wcb_weights_and_B(
            G_total=G_total,
            G_treated=G_treated,
            B_requested=self.config.wcb_B,
        )
        weights_pref = getattr(self.config, "wcb_weights", "auto")
        wcb_weights = wcb_weights_auto if weights_pref == "auto" else weights_pref

        cluster_spec = getattr(self.config, "cluster_col", None)
        if isinstance(cluster_spec, str):
            cluster_terms = [cluster_spec]
        else:
            cluster_terms = list(cluster_spec or [])
        if not cluster_terms:
            cluster_terms = ["unit_id"]

        wcb_args: Optional[Dict[str, Any]] = None
        if getattr(self.config, "use_wcb", True):
            wcb_args = {
                "weights": wcb_weights,
                "B": wcb_B,
                "cluster_spec": cluster_terms,
                "impose_null": getattr(self.config, "wcb_impose_null", True),
                "seed": getattr(self.config, "seed", None),
            }

        wcb_meta = {
            "weights": wcb_weights,
            "B": wcb_B,
            "cluster_spec": cluster_terms,
            "enabled": bool(wcb_args),
        }

        # 3) Estimator
        self._estimator = DidEstimator(self.config)
        result = DidStudyResult(config=self.config, data=panel)

        # 4) ATT^o (pooled)
        if run_att:
            result.att = self.estimator.estimate_att_o(panel=panel, wcb_args=wcb_args)

        # 5) Bins
        has_dose_bins = bool(df is not None and "dose_bin" in df.columns)
        if run_bins and has_dose_bins:
            result.bins = self.estimator.estimate_binned_att_o(panel=panel, wcb_args=wcb_args)

        # 6) Event study
        if run_event_study:
            result.event_study = self.estimator.event_study(panel=panel, wcb_args=wcb_args)
            if result.event_study is not None:
                result.es_wcb_p = getattr(result.event_study, "wcb_p", None)

        # 7) HonestDiD (Δ^RM)
        honestdid_enabled = bool(getattr(self.config, "honestdid_enable", False))
        if run_honest_did and honestdid_enabled:
            result.honest_did = self._run_honest_did(panel, result)

        # 8) Meta
        if result.es_wcb_p is not None:
            wcb_meta["es_wcb_p"] = result.es_wcb_p
        result.wcb_meta = wcb_meta
        return result

    # ------------------------------------------------------------------
    # HonestDiD (Δ^RM, Rambachan & Roth 2023)
    # ------------------------------------------------------------------
    def _run_honest_did(
        self,
        panel: PanelData,
        result: DidStudyResult,
    ) -> Dict[str, Any]:
        """
        Compute Rambachan & Roth (2023) Δ^RM bounds for a scalar parameter θ.

        θ is defined as a *linear functional* of the post-treatment event-study
        coefficients:

            θ = l_vec' * τ_post

        where τ_post is the vector of post-treatment event-time effects and
        l_vec are weights. We choose l_vec proportional to the number of
        treated observations in each post-event horizon (i.e. an exposure-
        weighted average ATT^O), falling back to uniform weights if needed.

        This θ is then passed as `l_vec` to HonestDiD's
        createSensitivityResults_relativeMagnitudes() so that the naive
        estimate θ_hat and the HonestDiD robust bounds refer to the *same*
        estimand.
        """
        es = result.event_study
        if es is None or es.coefs is None or es.coefs.empty:
            return {}

        names_pre = list(getattr(es, "names_pre", []) or [])
        names_post = list(getattr(es, "names_post", []) or [])
        if not names_pre or not names_post:
            return {}

        ordered_names = names_pre + names_post

        # Event-study coefficient table (from estimator.event_study)
        coef_tab = es.coefs  # columns typically: ['event_time', 'beta', 'se', ...]

        def beta_for(name: str) -> Optional[float]:
            tau = _tau_from_es_name(name)
            if tau is None:
                return None
            row = coef_tab.loc[coef_tab["event_time"] == tau]
            return None if row.empty else float(row.iloc[0]["beta"])

        betas = [beta_for(n) for n in ordered_names]
        if any(v is None for v in betas):
            # Incomplete mapping from ES_* names back to event_time
            return {}
        beta_s = pd.Series(betas, index=ordered_names, dtype=float)

        # Pre/post event times as integers
        pre_periods = [_tau_from_es_name(n) for n in names_pre]
        post_periods = [_tau_from_es_name(n) for n in names_post]
        if not pre_periods or not post_periods or any(
            x is None for x in pre_periods + post_periods
        ):
            return {}

        # ------------------------------------------------------------------
        # Construct l_vec (weights over post-treatment horizons)
        # ------------------------------------------------------------------
        used = getattr(es, "data", None)
        l_vec = None
        if used is not None and names_post:
            # Exposure-based weights: number of treated observations contributing
            # to each event-time dummy ES_tk in the design matrix used to fit
            # the event study.
            counts = []
            for nm in names_post:
                counts.append(float(used[nm].sum()) if nm in used.columns else 0.0)
            total = float(sum(counts))
            if total > 0.0:
                l_vec = np.array(counts, dtype=float) / total

        # ------------------------------------------------------------------
        # Mbar grid configuration
        # ------------------------------------------------------------------
        M_grid_cfg = getattr(self.config, "honestdid_M_grid", None)
        if M_grid_cfg:
            grid_sorted = sorted(float(x) for x in M_grid_cfg if x is not None)
            grid_points = max(len(grid_sorted), 10)
            Mmax = (
                float(grid_sorted[-1])
                if grid_sorted
                else getattr(self.config, "honestdid_Mbar", None)
            )
        else:
            grid_points = 10
            Mmax = getattr(self.config, "honestdid_Mbar", None)
        if Mmax is None:
            # Default upper bound for Δ^RM grid; substantive choice
            Mmax = 2.0

        # ------------------------------------------------------------------
        # Call HonestDiD (Δ^RM, method C-LF) via R
        # ------------------------------------------------------------------
        try:
            res = honest_did_bounds(
                es_df=coef_tab,
                pre_periods=pre_periods,
                post_periods=post_periods,
                beta_col="beta",
                se_col="se",
                Mmax=Mmax,
                grid_points=int(grid_points),
                seed=getattr(self.config, "seed", None),
                l_vec=l_vec,
            )
        except Exception as e:
            print(f"[HonestDiD] failed: {e}")
            return {}

        # ------------------------------------------------------------------
        # Build output: M-grid, bounds, and naive θ̂
        # ------------------------------------------------------------------
        M = np.array(res.M, dtype=float)
        lo = np.array(res.lb, dtype=float)
        hi = np.array(res.ub, dtype=float)

        # Naive θ̂ computed from the same l_vec used in HonestDiD
        post_betas = np.array(
            [beta_s.get(n, np.nan) for n in names_post if n in beta_s.index],
            dtype=float,
        )
        if post_betas.size == 0:
            return {}

        if l_vec is None or len(l_vec) != post_betas.size:
            # Fallback: uniform weighting over post periods
            weights = np.ones(post_betas.size, dtype=float) / post_betas.size
        else:
            weights = l_vec[:post_betas.size]

        theta_hat = float(np.dot(post_betas, weights))

        return {"M": M, "lo": lo, "hi": hi, "theta_hat": theta_hat}
