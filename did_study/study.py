# did_study/study.py
# COMPLETE VERSION - November 14, 2025

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
    """Orchestrates the full DiD analysis pipeline."""

    def __init__(self, config: StudyConfig) -> None:
        self.config = config
        self._estimator: Optional[DidEstimator] = None
        self._panel: Optional[PanelData] = None

    @property
    def estimator(self) -> DidEstimator:
        if self._estimator is None:
            raise RuntimeError("Estimator not initialised yet. Call .run().")
        return self._estimator

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

        Parameters
        ----------
        run_att : bool
            If True, estimate pooled ATT^o.
        run_bins : bool
            If True, estimate dose-bin heterogeneous effects.
        run_event_study : bool
            If True, estimate event-study coefficients with leads/lags.
        run_honest_did : bool
            If True, run HonestDiD sensitivity analysis.

        Returns
        -------
        DidStudyResult
            Container with all estimates and robustness checks.
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

    def _run_honest_did(
        self,
        panel: PanelData,
        result: DidStudyResult,
    ) -> Dict[str, Any]:
        """
        Compute Rambachan & Roth (2023) Δ^RM bounds for a scalar parameter θ.

        θ = l_vec' * τ_post

        where τ_post is the vector of post-treatment event-time effects and
        l_vec are exposure-weighted or uniform weights.
        """

        es = result.event_study
        if es is None or es.coefs is None or es.coefs.empty:
            print("[HonestDiD] Event study results not available")
            return {}

        # Extract pre and post names
        names_pre = list(getattr(es, "names_pre", []) or [])
        names_post = list(getattr(es, "names_post", []) or [])

        if not names_pre or not names_post:
            print(f"[HonestDiD] Insufficient pre ({len(names_pre)}) or post ({len(names_post)}) periods")
            return {}

        # Extract the full covariance matrix
        vcov = getattr(es, "vcov", None)
        if vcov is None or vcov.size == 0:
            print("[HonestDiD] No vcov matrix available from event study")
            return {}

        # =====================================================================
        # Extract betas in the correct [pre, post] order
        # =====================================================================
        coef_tab = es.coefs  # DataFrame with columns: event_time, beta, se, p

        def tau_from_name(name: str) -> Optional[int]:
            """Parse ES_t*/ES_tm* names to event times."""
            if name.startswith("ES_tm"):
                return -int(name.replace("ES_tm", ""))
            if name.startswith("ES_t"):
                return int(name.replace("ES_t", ""))
            return None

        # Get event times for pre and post
        pre_taus = [tau_from_name(n) for n in names_pre]
        post_taus = [tau_from_name(n) for n in names_post]

        if any(t is None for t in pre_taus + post_taus):
            print("[HonestDiD] Could not parse all event-time names")
            return {}

        # Extract beta for each event time
        def beta_for_tau(tau: int) -> Optional[float]:
            row = coef_tab.loc[coef_tab["event_time"] == tau]
            return None if row.empty else float(row.iloc[0]["beta"])

        pre_betas = [beta_for_tau(t) for t in pre_taus]
        post_betas = [beta_for_tau(t) for t in post_taus]

        if any(b is None for b in pre_betas + post_betas):
            print("[HonestDiD] Missing beta values for some event times")
            return {}

        betas = np.array(pre_betas + post_betas, dtype=float)

        # Validate vcov dimensions match betas
        expected_len = len(pre_taus) + len(post_taus)
        if vcov.shape != (expected_len, expected_len):
            print(
                f"[HonestDiD] vcov shape {vcov.shape} doesn't match betas length {expected_len}"
            )
            return {}

        # =====================================================================
        # Construct l_vec (weights over post-treatment horizons ONLY)
        # =====================================================================
        used = getattr(es, "data", None)
        l_vec = None

        if used is not None and names_post:
            # Exposure-based weights: number of treated observations at each event time
            counts = []
            for nm in names_post:
                counts.append(float(used[nm].sum()) if nm in used.columns else 0.0)

            total = float(sum(counts))

            if total > 0.0:
                l_vec = np.array(counts, dtype=float) / total
                print(f"[HonestDiD] Exposure-weighted l_vec: {l_vec}")
            else:
                # Fallback to uniform if no exposure data
                l_vec = np.ones(len(names_post), dtype=float) / len(names_post)
                print(f"[HonestDiD] No exposure data; using uniform l_vec: {l_vec}")
        else:
            # Fallback to uniform weights
            l_vec = np.ones(len(names_post), dtype=float) / len(names_post)
            print(f"[HonestDiD] Using uniform l_vec: {l_vec}")

        # Validate l_vec length
        if len(l_vec) != len(post_taus):
            print(
                f"[HonestDiD] l_vec length {len(l_vec)} != numPostPeriods {len(post_taus)}"
            )
            return {}

        # =====================================================================
        # Mbar grid configuration
        # =====================================================================
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
            Mmax = 2.0

        # =====================================================================
        # Call HonestDiD (Δ^RM, method C-LF) via R
        # =====================================================================
        try:
            res = honest_did_bounds(
                betas=betas,
                Sigma=vcov,  # Full covariance from event study
                numPrePeriods=len(pre_taus),
                numPostPeriods=len(post_taus),
                Mmax=Mmax,
                grid_points=int(grid_points),
                seed=getattr(self.config, "seed", None),
                l_vec=l_vec,
            )
        except Exception as e:
            print(f"[HonestDiD] failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

        # =====================================================================
        # Build output: M-grid, bounds, and naive θ̂
        # =====================================================================
        M = np.array(res.M, dtype=float)
        lo = np.array(res.lb, dtype=float)
        hi = np.array(res.ub, dtype=float)

        # Naive θ̂ computed from the same l_vec used in HonestDiD
        post_betas_arr = np.array(post_betas, dtype=float)
        theta_hat = float(np.dot(post_betas_arr, l_vec))

        return {
            "M": M,
            "lo": lo,
            "hi": hi,
            "theta_hat": theta_hat,
            "l_vec": l_vec,
            "numPrePeriods": len(pre_taus),
            "numPostPeriods": len(post_taus),
            "method": res.method,
            "delta_label": res.delta_label,
        }
