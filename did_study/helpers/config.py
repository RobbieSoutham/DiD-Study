# config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal
import pandas as pd

@dataclass
class StudyConfig:
    # =========================
    # Core data
    # =========================
    df: pd.DataFrame

    # =========================
    # Mapping & weights
    # =========================
    mapping: Optional[Dict[str, Any]] = None
    mapping_weights: Optional[Dict[str, Dict[str, float]]] = None

    # =========================
    # Outcome handling
    # =========================
    # Use "direct" (allocate emissions to CCUS sectors) or "total" (country total).
    # You may also pass the name of a custom outcome column in your df.
    outcome_mode: Literal["direct", "total"] | str = "direct"
    use_log_outcome: bool = True
    differenced: bool = True

    # =========================
    # Unit & schema
    # =========================
    unit_cols: Tuple[str, str] = ("Country", "CCUS_sector")
    year_col: str = "Year"
    outcome_col: str = "Emissions"
    emissions_sector_col: str = "emissions_sector"
    capacity_col: str = "eor_capacity"
    sector_col: str = "Sector"

    # =========================
    # Covariates & engineering
    # =========================
    supdem_mode: Literal["sum", "direct"] = "sum"
    covariates: Optional[List[str]] = None
    # If True and differenced=True, use L1 levels for covariates (bad-controls safe)
    use_lag_levels_in_diff: bool = False

    # Leakage-safe transforms fit set
    fit_on: Literal["never_or_notyet", "controls_only"] = "never_or_notyet"

    # =========================
    # Support / trimming
    # =========================
    min_pre: int = 2
    min_post: int = 1
    treat_threshold: float = 0.0  # tau

    # =========================
    # Dose binning
    # =========================
    # Option A: explicit edges
    dose_bins: Optional[List[float]] = None
    dose_bins_right_closed: bool = False
    # Option B: explicit quantiles in [0,1], e.g. [0, 1/3, 2/3, 1]
    dose_quantiles: Optional[List[float]] = None
    # Option C: convenience - set n_bins and we compute equal-frequency bins
    # on post-treated, positive dose. (Handled in PanelData._prepare)
    n_bins: Optional[int] = None

    # =========================
    # Artifacts
    # =========================
    artifact_dir: Optional[str] = None

    # =========================
    # Inference options (used by estimators/study)
    # =========================
    # Event study window (interactions)
    pre: int = 5
    post: int = 10
    min_cluster_support: int = 2
    cluster_col : list[str] = ("Country", "CCUS_sector")

    # Wild Cluster Bootstrap (WCB) selector
    use_wcb: bool = True
    wcb_weights: Literal["auto", "rademacher", "webb"] = "auto"
    wcb_B: Optional[int] = None
    wcb_impose_null: bool = True  # kept for completeness if needed downstream
    seed: int = 123

    # HonestDiD options (R-bridge; set *_enable to False to skip)
    honestdid_enable: bool = True           # pooled
    honestdid_by_bin: bool = False          # per-bin (can switch on later)
    honestdid_Mbar: Optional[float] = None  # if None, will calibrate/conservative
    honestdid_M_grid: Optional[List[float]] = None  # e.g. [0.0, 0.25, 0.5, 1.0]

    # R bridge (for HonestDiD / WCB original implementations)
    r_bridge: bool = True
    rscript_path: Optional[str] = None          # if None, rely on PATH
    r_lib_paths: Optional[List[str]] = None     # optional .libPaths()
    r_env: Optional[Dict[str, str]] = None      # env vars for subprocess

    def copy(self) -> "StudyConfig":
        # Shallow copy (df is shared by design); lists/dicts are defensively copied
        return StudyConfig(
            df=self.df,
            mapping=self.mapping,
            mapping_weights=self.mapping_weights,
            outcome_mode=self.outcome_mode,
            use_log_outcome=self.use_log_outcome,
            differenced=self.differenced,
            unit_cols=self.unit_cols,
            year_col=self.year_col,
            outcome_col=self.outcome_col,
            emissions_sector_col=self.emissions_sector_col,
            capacity_col=self.capacity_col,
            sector_col=self.sector_col,
            supdem_mode=self.supdem_mode,
            covariates=list(self.covariates) if self.covariates is not None else None,
            use_lag_levels_in_diff=self.use_lag_levels_in_diff,
            fit_on=self.fit_on,
            min_pre=self.min_pre,
            min_post=self.min_post,
            treat_threshold=self.treat_threshold,
            dose_bins=list(self.dose_bins) if self.dose_bins is not None else None,
            dose_bins_right_closed=self.dose_bins_right_closed,
            dose_quantiles=list(self.dose_quantiles) if self.dose_quantiles is not None else None,
            n_bins=self.n_bins,
            artifact_dir=self.artifact_dir,
            pre=self.pre,
            post=self.post,
            min_cluster_support=self.min_cluster_support,
            use_wcb=self.use_wcb,
            wcb_weights=self.wcb_weights,
            wcb_B=self.wcb_B,
            wcb_impose_null=self.wcb_impose_null,
            seed=self.seed,
            honestdid_enable=self.honestdid_enable,
            honestdid_by_bin=self.honestdid_by_bin,
            honestdid_Mbar=self.honestdid_Mbar,
            honestdid_M_grid=list(self.honestdid_M_grid) if self.honestdid_M_grid is not None else None,
            r_bridge=self.r_bridge,
            rscript_path=self.rscript_path,
            r_lib_paths=list(self.r_lib_paths) if self.r_lib_paths is not None else None,
            r_env=dict(self.r_env) if self.r_env is not None else None,
        )
