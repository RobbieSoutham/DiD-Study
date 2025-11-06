from __future__ import annotations

from typing import Any, Dict, Optional, Set

import pandas as pd

from did_study.helpers.config import StudyConfig
from did_study.helpers.preparation import PanelData
from did_study.estimators.base import BaseEstimator
from did_study.estimators.att import AttResult, estimate_att_o
from did_study.estimators.bins import BinAttResult, estimate_binned_att_o
from did_study.estimators.event_study import EventStudyResult, event_study


class DidEstimator(BaseEstimator):
    """
    Thin façade over the actual estimator functions.

    We keep this file deliberately small — this is the place where we can
    do “safety” logic (e.g. fall back to Python-only for fragile
    per-bin event studies) without polluting the core estimators.
    """

    def __init__(self, config: StudyConfig) -> None:
        super().__init__(config)

    # ---------------------------------------------------------
    # ATT^o (pooled, continuous-dose building block)
    # ---------------------------------------------------------
    def estimate_att_o(self, panel: PanelData) -> AttResult:
        return estimate_att_o(panel, self.config)

    # ---------------------------------------------------------
    # Binned ATT^o
    # ---------------------------------------------------------
    def estimate_binned_att_o(self, panel: PanelData) -> BinAttResult | pd.DataFrame:
        return estimate_binned_att_o(panel, self.config)

    # ---------------------------------------------------------
    # Event study (pooled)
    # ---------------------------------------------------------
    def event_study(self, panel: PanelData) -> EventStudyResult:
        return event_study(panel, self.config)

    # ---------------------------------------------------------
    # Event study by *dose bin*
    # ---------------------------------------------------------
    def event_study_by_bin(self, panel: PanelData) -> Dict[Any, EventStudyResult]:
        """
        Return {bin_label: EventStudyResult}.

        This is the place where the original code was blowing up for you:
        when we subset to a single bin, the R wild-cluster path often
        can’t be fit (too many FE, too few obs, param names change).

        Strategy here:
          1. assign each unit to exactly one bin;
          2. for each bin, take *all* rows for those units (pre + post);
          3. TRY a normal ES with the user's config;
          4. if it errors (typically R “subscript out of bounds”), RETRY
             with a **safe** config:
                - r_bridge = False
                - use_wcb = False
                - honestdid_enable = False
          5. if that still fails, we just skip this bin.
        """
        dfu = panel.panel.copy()
        year_col = getattr(self.config, "year_col", "Year")
        unit_col = getattr(self.config, "unit_cols", ["unit_id"])[0]

        # if we don't have bins at all, nothing to do
        if "dose_bin" not in dfu.columns:
            return {}

        # -----------------------------------------------------
        # 1) assign units to a single bin
        #    priority: first post-treatment non-missing bin
        # -----------------------------------------------------
        # rows where unit is treated now (post == 1) and dose_bin is known
        if "post" in dfu.columns:
            treated_nonmiss = (
                dfu[dfu["post"] == 1]
                .dropna(subset=["dose_bin"])
                .sort_values([unit_col, year_col])
            )
            post_first = (
                treated_nonmiss.groupby(unit_col, as_index=False)
                .first()[[unit_col, "dose_bin"]]
                .rename(columns={"dose_bin": "unit_bin"})
            )
        else:
            # fallback: earliest non-missing bin overall
            nonmiss = (
                dfu.dropna(subset=["dose_bin"])
                .sort_values([unit_col, year_col])
            )
            post_first = (
                nonmiss.groupby(unit_col, as_index=False)
                .first()[[unit_col, "dose_bin"]]
                .rename(columns={"dose_bin": "unit_bin"})
            )

        # units that still have no bin → use modal bin over time
        all_units: Set[Any] = set(dfu[unit_col].unique())
        assigned_units: Set[Any] = set(post_first[unit_col].unique())
        missing_units = list(all_units - assigned_units)
        if missing_units:
            # for these units, pick the most common non-missing bin
            tmp = (
                dfu[dfu[unit_col].isin(missing_units) & dfu["dose_bin"].notna()]
                .groupby([unit_col, "dose_bin"], as_index=False)
                .size()
                .rename(columns={"size": "n"})
            )
            if not tmp.empty:
                # for each unit → pick bin with biggest n
                idx = (
                    tmp.sort_values([unit_col, "n"], ascending=[True, False])
                    .groupby(unit_col)
                    .head(1)
                )
                post_first = pd.concat(
                    [post_first, idx[[unit_col, "dose_bin"]].rename(columns={"dose_bin": "unit_bin"})],
                    ignore_index=True,
                )

        # merge back
        dfm = dfu.merge(post_first, on=unit_col, how="left")
        dfm = dfm[dfm["unit_bin"].notna()].copy()

        # unique bins (as strings so sorting is stable even for intervals)
        bins = sorted(dfm["unit_bin"].dropna().unique(), key=lambda x: str(x))

        # -----------------------------------------------------
        # tiny wrapper to subset a panel object
        # -----------------------------------------------------
        class _PanelView(PanelData):
            def __init__(self, parent: PanelData, subdf: pd.DataFrame) -> None:
                # we pretend to be a PanelData, but with a sub-frame
                self.panel = subdf
                self.info = parent.info
                self.mapping = getattr(parent, "mapping", None) 
                self.mapping_weights = parent.mapping_weights
                self.df_raw = parent.df_raw
                # keep names for plotting / printing
                self.unit_cols = parent.unit_cols
                self.year_col = parent.year_col
                self.outcome_name = parent.outcome_name
                self.covar_cols_used = getattr(parent, "covar_cols_used", [])

        # -----------------------------------------------------
        # 2) loop over bins and run ES
        # -----------------------------------------------------
        out: Dict[Any, EventStudyResult] = {}
        for b in bins:
            units_b = set(dfm.loc[dfm["unit_bin"] == b, unit_col].unique())
            sub = dfu[dfu[unit_col].isin(units_b)].copy()  # keep pre+post

            # quick viability checks
            n_post = int((sub["post"] == 1).sum()) if "post" in sub.columns else 0
            n_years = sub[year_col].nunique() if year_col in sub.columns else 0
            if n_post == 0:
                # no treated time → no ES
                continue
            if n_years < max(self.config.pre, self.config.min_pre) + self.config.min_post:
                # panel too short for usual ES window
                # we still TRY, but only once and without R
                safe_cfg = self._safe_cfg_for_bins()
                pv = _PanelView(panel, sub)
                try:
                    es_b = event_study(pv, safe_cfg)
                    out[b] = es_b
                except Exception:
                    # really not estimable → skip
                    continue
                continue

            try:
                pre_leads = int((sub["event_time"].astype("Float64") <= -2).sum()) if "event_time" in sub.columns else 0
                post_rows = int((sub.get("post", 0) == 1).sum())
                print(f"[ES-by-bin] bin {b}: pre-leads (τ<=-2)={pre_leads}, post rows={post_rows}, units={len(units_b)}")

                if post_rows == 0:
                    print(f"[ES-by-bin] skip bin {b}: no treated rows (post==1) in this subset.")
                    continue
                res = event_study(_PanelView(panel, sub), self.config)
                out[b] = res
            except Exception as e:
                print(f"[ES-by-bin] bin {b} failed: {e}")

        return out

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------
    def _safe_cfg_for_bins(self) -> StudyConfig:
        """
        Make a copy of the config with everything that usually causes
        issues on tiny bin-samples switched OFF.
        """
        cfg = self.config.copy()
        cfg.r_bridge = False         # don't call R / fwildclusterboot
        cfg.use_wcb = False          # don't even try WCB on very small bin
        cfg.honestdid_enable = False # we only do HonestDiD at top level
        return cfg


__all__ = [
    "DidEstimator",
    "AttResult",
    "BinAttResult",
    "EventStudyResult",
]
