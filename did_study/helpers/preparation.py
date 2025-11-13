# did_study/helpers/preparation.py
from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .config import StudyConfig
from ..estimators.bins import make_dose_bins
from .defaults import DEFAULT_MAPPING

_SECTORAL_PREFIXES = ("Supply_", "Demand_")


# ----------------------------
# Basic helpers
# ----------------------------
def _em_map_is_partition(emap: Dict[str, set]) -> bool:
    """True iff each emissions sector maps to exactly one CCUS sector."""
    seen: Dict[str, str] = {}
    for ccus, es in (emap or {}).items():
        for e in set(es):
            if e in seen and seen[e] != ccus:
                return False
            seen[e] = ccus
    return True


def _build_supdem_cy(
    df: pd.DataFrame, *, sector_col: str, country_col: str, year_col: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Pivot sectoral Supply_/Demand_ series to country-year 'wide' columns and add *_total."""
    base_vars = [
        c for c in df.columns
        if c.startswith(_SECTORAL_PREFIXES) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not base_vars:
        return df[[country_col, year_col]].drop_duplicates(), []

    long = df[[country_col, year_col, sector_col] + base_vars].melt(
        id_vars=[country_col, year_col, sector_col],
        var_name="base", value_name="val"
    )
    long["wide_col"] = long["base"] + "__" + long[sector_col].astype(str)

    wide = (
        long.pivot_table(index=[country_col, year_col],
                         columns="wide_col", values="val", aggfunc="sum")
            .reset_index()
    )
    wide_cols = [c for c in wide.columns if isinstance(c, str) and "__" in c]

    for b in sorted(set(long["base"])):
        cols_b = [c for c in wide_cols if c.startswith(b + "__")]
        if cols_b:
            wide[b + "_total"] = wide[cols_b].sum(axis=1, min_count=1)

    return wide, wide_cols


def _sum_cols_block(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Sum all wide columns that start with <prefix>__ ; fallback to <prefix>_total; else NaNs."""
    cols = [c for c in df.columns if c.startswith(prefix + "__")]
    if not cols:
        tot = prefix + "_total"
        cols = [tot] if tot in df.columns else []
    return df[cols].sum(axis=1, min_count=1) if cols else pd.Series(np.nan, index=df.index)


# ----------------------------
# Panel builder
# ----------------------------
class PanelData:
    """
    Country(/CCUS) x Year panel with:
      - absorbing adoption (once treated, always treated),
      - CY Supply/Demand merged after adoption timing is built,
      - PCA/covariates fit on pre rows only (never- or not-yet-treated),
      - absorbing dose bins (cohort = first post bin),
      - optional direct allocation of emissions to CCUS sectors,
      - leakage-safe transformations and scaling.

    Use via `prepare_ccus_panel(StudyConfig)`.
    """

    def __init__(self, config: StudyConfig) -> None:
        self.config = config.copy()
        self.panel: Optional[pd.DataFrame] = None
        self.info: Dict[str, Any] = {}
        self.outcome_name: Optional[str] = None
        self.covar_cols_used: List[str] = []
        self._prepare()

    # ---------- emissions allocation for outcome_mode="direct"
    def _alloc_emissions_to_ccus(
        self,
        d: pd.DataFrame,
        *,
        country_col: str,
        year_col: str,
        emissions_sector_col: str,
        outcome_col: str,
        ccus_col: str,
        emap: Dict[str, set],
        weights: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        base_em = (
            d[[country_col, year_col, emissions_sector_col, outcome_col]]
            .dropna(subset=[outcome_col])
            .groupby([country_col, year_col, emissions_sector_col], as_index=False)[outcome_col]
            .sum()
        )

        if weights:
            rows = []
            for e, grp in base_em.groupby(emissions_sector_col):
                wmap = (weights or {}).get(e, {})
                if not wmap:
                    continue
                for ccus, w in wmap.items():
                    if w is None or w < 0:
                        continue
                    tmp = grp.copy()
                    tmp[ccus_col] = ccus
                    tmp[outcome_col] = tmp[outcome_col] * float(w)
                    rows.append(tmp)
            if not rows:
                raise ValueError("[direct] Provided 'mapping_weights' produced no allocation rows.")
            alloc = pd.concat(rows, ignore_index=True)
        else:
            if not _em_map_is_partition(emap):
                raise ValueError("[direct] 'emissions_to_ccus' must be a partition or provide 'mapping_weights'.")
            e_to_ccus = {e: ccus for ccus, S in (emap or {}).items() for e in set(S)}
            alloc = base_em.copy()
            alloc[ccus_col] = alloc[emissions_sector_col].map(e_to_ccus).astype(object)
            alloc = alloc[alloc[ccus_col].notna()].copy()

        y_ccus = (
            alloc.groupby([country_col, ccus_col, year_col], as_index=False)[outcome_col]
            .sum()
            .rename(columns={outcome_col: "y_level"})
        )
        return y_ccus

    # ---------- absorbing dose bins (cohort = first post bin)
    def _bin_dose_absorbing(
        self,
        g: pd.DataFrame,
        cfg: StudyConfig,
    ) -> tuple[pd.DataFrame, Optional[dict]]:
        edges_used = None
        method_used = None

        # (a) explicit numeric edges
        if getattr(cfg, "dose_bins", None):
            ed = np.asarray(cfg.dose_bins, float)
            if ed.size >= 2:
                edges_used = np.unique(ed)
                method_used = "edges"

        # (b) explicit quantiles
        elif getattr(cfg, "dose_quantiles", None):
            qs = np.asarray(cfg.dose_quantiles, float)
            qs = qs[(qs >= 0) & (qs <= 1)]
            base = g.loc[(g["post"] == 1) & (g["dose_level"] > 0), "dose_level"].dropna().values
            if base.size and qs.size >= 2:
                ed = np.unique(np.quantile(base, qs))
                if ed.size >= 2:
                    edges_used = ed
                    method_used = "quantiles"

        # (c) convenience: n_bins -> quantile edges
        elif hasattr(cfg, "n_bins") and getattr(cfg, "n_bins", None):
            try:
                n = int(cfg.n_bins)
            except Exception:
                n = 0
            if n >= 2:
                base = g.loc[(g["post"] == 1) & (g["dose_level"] > 0), "dose_level"].dropna().values
                if base.size:
                    ed = np.unique(np.quantile(base, np.linspace(0.0, 1.0, n + 1)))
                    if ed.size >= 2:
                        edges_used = ed
                        method_used = "n_bins_quantile"

        # no binning requested/possible
        if edges_used is None:
            return g, None

        # Use shared make_dose_bins to build human-readable, ordered labels
        raw_c = make_dose_bins(
            g,
            dose_col="dose_level",
            edges=edges_used,
            quantiles=None,
            include_untreated=True,
            untreated_label="untreated",
            right=getattr(cfg, "dose_bins_right_closed", False),
            precision=2,
         )
        tmp = g.assign(_raw_bin=raw_c)

        # first post-treatment bin per unit (absorbing)
        first_bin = (
            tmp.loc[tmp["post"] == 1, ["unit_id", "g", "_raw_bin"]]
               .dropna(subset=["_raw_bin"])
               .sort_values(["unit_id", "g"])
               .groupby("unit_id", as_index=False)
               .first()
               .rename(columns={"_raw_bin": "bin_cohort"})
        )
        g = g.merge(first_bin[["unit_id", "bin_cohort"]], on="unit_id", how="left")

        # Assign bin only to post rows; pre rows keep NaN
        g["dose_bin"] = np.where(g["post"] == 1, g["bin_cohort"], pd.NA)

        # Support diagnostics (post rows only)
        post_rows = g.loc[g["post"] == 1]
        sup_units = post_rows.groupby("dose_bin")["unit_id"].nunique().rename("units")
        sup_rows = post_rows.groupby("dose_bin")["unit_id"].size().rename("rows")
        support = {
            "method": method_used,
            "edges": (edges_used.tolist() if isinstance(edges_used, np.ndarray) else list(edges_used)),
            "units": sup_units.to_dict(),
            "rows": sup_rows.to_dict(),
        }

        print(f"[prepare][bins] absorbing ({method_used}); edges={support['edges']}")
        for b in sorted([x for x in post_rows["dose_bin"].dropna().unique()], key=str):
            if str(b) == "untreated":
                continue
            u = int(sup_units.get(b, 0))
            r = int(sup_rows.get(b, 0))
            doses = g.loc[(g["dose_bin"] == b) & (g["post"] == 1), "dose_level"]
            if doses.empty:
                print(f"  - {b}: units={u}, rows={r}")
                continue
            mean_dose = doses.mean()
            cv = (doses.std() / mean_dose) if mean_dose not in (0, np.nan) else np.nan
            msg = f"  - {b}: units={u}, rows={r}, dose range=[{doses.min():.1f}, {doses.max():.1f}]"
            if np.isfinite(cv):
                msg += f", CV={cv:.2f}"
            print(msg)

        return g, support

    # ---------- main builder
    def _prepare(self) -> None:
        cfg = self.config
        d = cfg.df.copy()

        # required columns
        need = {
            cfg.unit_cols[0], cfg.unit_cols[1], cfg.sector_col,
            cfg.emissions_sector_col, cfg.year_col, cfg.outcome_col,
            cfg.capacity_col
        }
        miss = need - set(d.columns)
        if miss:
            raise ValueError(f"Missing columns in data: {sorted(miss)}")

        mapping = getattr(cfg, "mapping", None) or DEFAULT_MAPPING
        self.mapping = mapping
        country_col, ccus_col = cfg.unit_cols

        # ---------- (0) Base unit grid & dose ----------
        if cfg.outcome_mode == "total":
            grid = d[[country_col, cfg.year_col]].drop_duplicates()
            cap_source = d[[country_col, cfg.year_col, cfg.capacity_col]].drop_duplicates()
            cap = (
                cap_source.groupby([country_col, cfg.year_col], as_index=False)[cfg.capacity_col]
                    .sum()
                    .rename(columns={cfg.capacity_col: "dose_level"})
            )
            g = (
                grid.merge(cap, on=[country_col, cfg.year_col], how="left")
                    .sort_values([country_col, cfg.year_col])
            )
            g["unit_id"] = g[country_col].astype(str)
            unit_level = "country"
        else:
            grid = d[[country_col, ccus_col, cfg.year_col]].drop_duplicates()
            cap_source = d[[country_col, ccus_col, cfg.year_col, cfg.capacity_col]].drop_duplicates()
            cap = (
                cap_source.groupby([country_col, ccus_col, cfg.year_col], as_index=False)[cfg.capacity_col]
                    .sum()
                    .rename(columns={cfg.capacity_col: "dose_level"})
            )
            g = (
                grid.merge(cap, on=[country_col, ccus_col, cfg.year_col], how="left")
                    .sort_values([country_col, ccus_col, cfg.year_col])
            )
            g["unit_id"] = g[country_col].astype(str) + "_" + g[ccus_col].astype(str)
            unit_level = "sector"

        g["dose_level"] = g["dose_level"].fillna(0.0)
        g["treated_now"] = (g["dose_level"] > cfg.treat_threshold).astype(int)

        # ---------- (1) Outcome (y_level -> log/Deltalog or level/Delta) ----------
        used_outcome_mode = cfg.outcome_mode
        if cfg.outcome_mode == "direct":
            emap = (mapping or {}).get("emissions_to_ccus", {})
            if not _em_map_is_partition(emap) and getattr(cfg, "mapping_weights", None) is None:
                raise ValueError("[prepare] emissions_to_ccus mapping must be a partition or provide mapping_weights.")
            if unit_level != "sector":
                raise ValueError("[prepare] direct mode requires sector units (CountryxCCUS_sector).")
            y_ccus = self._alloc_emissions_to_ccus(
                d,
                country_col=country_col, year_col=cfg.year_col,
                emissions_sector_col=cfg.emissions_sector_col, outcome_col=cfg.outcome_col,
                ccus_col=ccus_col, emap=emap, weights=getattr(cfg, "mapping_weights", None)
            )
            g = g.merge(y_ccus, on=[country_col, ccus_col, cfg.year_col], how="left")
        elif cfg.outcome_mode == "total":
            y_cy = (
                d[[country_col, cfg.year_col, cfg.emissions_sector_col, cfg.outcome_col]]
                .dropna(subset=[cfg.outcome_col])
                .groupby([country_col, cfg.year_col], as_index=False)[cfg.outcome_col]
                .sum()
                .rename(columns={cfg.outcome_col: "y_level"})
            )
            g = g.merge(y_cy, on=[country_col, cfg.year_col], how="left")
        else:
            # custom outcome column name in df
            if cfg.outcome_mode not in d.columns:
                raise ValueError(f"[prepare] outcome mode '{cfg.outcome_mode}' not found in df.")
            g = g.merge(
                d[[country_col, ccus_col, cfg.year_col, cfg.outcome_mode]].drop_duplicates()
                if unit_level == "sector" else
                d[[country_col, cfg.year_col, cfg.outcome_mode]].drop_duplicates(),
                on=[country_col, ccus_col, cfg.year_col] if unit_level == "sector" else [country_col, cfg.year_col],
                how="left"
            )
            g = g.rename(columns={cfg.outcome_mode: "y_level"})
            used_outcome_mode = "custom"

        if cfg.outcome_mode == "total" and unit_level == "sector":
            raise ValueError("Outcome is country-total but units are CountryxCCUS_sector.")

        # outcome transform
        g = g.sort_values(["unit_id", cfg.year_col]).copy()
        g = g[~g["y_level"].isna()].copy()
        eps = 1e-9
        if cfg.use_log_outcome:
            g["log_y"] = np.log(g["y_level"].clip(lower=0) + eps)
            if cfg.differenced:
                g["d_log_y"] = g.groupby("unit_id", sort=False)["log_y"].diff()
                outcome_name = "d_log_y"
            else:
                outcome_name = "log_y"
        else:
            if cfg.differenced:
                g["d_y"] = g.groupby("unit_id", sort=False)["y_level"].diff()
                outcome_name = "d_y"
            else:
                outcome_name = "y_level"
        g = g.dropna(subset=[outcome_name]).copy()
        self.outcome_name = outcome_name

        # ---------- (2) Absorbing adoption & event time ----------
        g["adopt_now"] = (g["dose_level"] > cfg.treat_threshold).astype(int)
        g["treated_ever"] = g.groupby("unit_id")["adopt_now"].transform("max")
        first_treat = (
            g.loc[g["adopt_now"] == 1]
             .groupby("unit_id", as_index=False)[cfg.year_col]
             .min()
             .rename(columns={cfg.year_col: "g"})
        )
        g = g.merge(first_treat, on="unit_id", how="left")
        g["post"] = ((g["treated_ever"] == 1) & (g[cfg.year_col] >= g["g"])).astype(int)
        g["event_time"] = np.where(
            g["treated_ever"] == 1, (g[cfg.year_col] - g["g"]).astype("Int64"), pd.NA
        )

        # ---------- (2a) Absorbing dose bins (BEFORE covariate preprocessing) ----------
        # Build bins using original dose units and store support info
        g, dose_bin_support = self._bin_dose_absorbing(g, cfg)

        # ---------- (3) Merge Supply/Demand (CY) ----------
        sd_cy, sd_wide_names = _build_supdem_cy(
            d, sector_col=cfg.sector_col, country_col=country_col, year_col=cfg.year_col
        )
        g = g.merge(sd_cy, on=[country_col, cfg.year_col], how="left")

        # ---------- (4) Optional S/D aggregation by CCUS + composites ----------
        if cfg.outcome_mode != "total" and unit_level == "sector":
            if getattr(cfg, "supdem_mode", "sum") == "direct":
                sdmap = (mapping or {}).get("ccus_to_supdem", {})
                bases = sorted({
                    c.split("__")[0]
                    for c in g.columns
                    if "__" in c and c.startswith(("Demand_", "Supply_"))
                })
                for b in bases:
                    out_col = b
                    g[out_col] = np.nan
                    for ccus, sec_list in (sdmap or {}).items():
                        m = (g[ccus_col] == ccus)
                        if not m.any():
                            continue
                        cols = [f"{b}__{s}" for s in (sec_list or []) if f"{b}__{s}" in g.columns]
                        if cols:
                            g.loc[m, out_col] = g.loc[m, cols].sum(axis=1, min_count=1)
                    tot = b + "_total"
                    if tot in g.columns:
                        g[out_col] = g[out_col].where(~g[out_col].isna(), g[tot])

        # Composites (levels)
        sup_ren = _sum_cols_block(g, "Supply_renewable_sources")
        sup_fos = _sum_cols_block(g, "Supply_fossil_fuels")
        eps_unit = (
            sup_fos.groupby(g["unit_id"]).transform(
                lambda s: max(1e-9, 1e-6 * float(np.nanmedian(s[s > 0])) if np.any(s > 0) else 1e-6)
            )
        )
        g["renewable_to_fossil_supply_ratio"] = (sup_ren + eps_unit) / (sup_fos + eps_unit)

        # Fuel mix shares - build nuclear share and its lag/diff unconditionally
        sup_nuc = _sum_cols_block(g, "Supply_nuclear")
        mix_den = sup_nuc + sup_ren + sup_fos
        eps_mix = (
            mix_den.groupby(g["unit_id"])
            .transform(lambda s: max(1e-12, float(np.nanmedian(s[s > 0])) if np.any(s > 0) else 1e-12))
        )
        g["nuclear_share_supply"] = (sup_nuc) / (mix_den + eps_mix)
        g["nuclear_share_supply"] = g["nuclear_share_supply"].clip(lower=0.0, upper=1.0)
        # Provide both lag and diff forms so users can reference directly
        g["L1_nuclear_share_supply"] = g.groupby("unit_id", sort=False)["nuclear_share_supply"].shift(1)
        g["d_nuclear_share_supply"] = g.groupby("unit_id", sort=False)["nuclear_share_supply"].diff()

        # ---------- (5) Macro covariates (CY) requested by user ----------
        requested_raw = list(dict.fromkeys(cfg.covariates or []))

        def _base_name(x: str) -> str:
            return x[2:] if x.startswith("d_") else (x[3:] if x.startswith("L1_") else x)

        requested_bases = [_base_name(x) for x in requested_raw]
        macro_candidates = [
            b for b in requested_bases
            if not b.startswith(("Supply_", "Demand_")) and b not in g.columns
        ]
        present_in_df = [c for c in macro_candidates if c in d.columns]
        if present_in_df:
            def first_valid(series):
                for v in series:
                    if pd.notna(v):
                        return v
                return np.nan
            macro_cy = (
                d[[country_col, cfg.year_col] + present_in_df]
                .groupby([country_col, cfg.year_col], as_index=False)
                .agg(first_valid)
            )
            g = g.merge(macro_cy, on=[country_col, cfg.year_col], how="left")

        # ---------- (6) PCA (pre-only) + covariate resolution & scaling ----------
        pca_info = {"scaler_path": None, "model_path": None, "pc1_explained": np.nan, "vars_used": []}
        pca_bases = [
            "Demand_natural_gas",
            "Demand_oil_products",
            "Demand_coal_peat_and_oil_shale",
            "Demand_crude_ngl_and_feedstocks",
        ]
        for base in pca_bases:
            cname = f"{base}__pca_base"
            g[cname] = _sum_cols_block(g, base)
        pca_cols = [f"{b}__pca_base" for b in pca_bases]

        if any(c in g.columns for c in pca_cols):
            fit_on = getattr(cfg, "fit_on", "never_or_notyet")
            pre_mask = (
                (g["treated_ever"] == 0)
                if (fit_on == "controls_only")
                else ((g["treated_ever"] == 0) | (g[cfg.year_col] < g["g"]))
            )
            Xp = g.loc[pre_mask, pca_cols].copy()
            if Xp.notna().any().any():
                for c in pca_cols:
                    med = Xp[c].median(skipna=True)
                    Xp[c] = Xp[c].fillna(med)
                pca_scaler = StandardScaler()
                _ = pca_scaler.fit_transform(Xp.values)
                pca = PCA(n_components=1, svd_solver="full", random_state=0)
                _ = pca.fit_transform(pca_scaler.transform(Xp.values))

                X_all = g[pca_cols].copy()
                for c in pca_cols:
                    med = Xp[c].median(skipna=True)
                    X_all[c] = X_all[c].fillna(med)
                g["energy_demand_fossil_fuels"] = pca.transform(
                    pca_scaler.transform(X_all.values)
                )[:, 0]
                pca_info["vars_used"] = pca_bases
                pca_info["pc1_explained"] = float(pca.explained_variance_ratio_[0])

                if getattr(cfg, "artifact_dir", None):
                    Path(cfg.artifact_dir).mkdir(parents=True, exist_ok=True)
                    import joblib as _joblib
                    _joblib.dump(pca_scaler, Path(cfg.artifact_dir) / "energy_pca_scaler.joblib")
                    _joblib.dump(pca,        Path(cfg.artifact_dir) / "energy_pca.joblib")
            else:
                g["energy_demand_fossil_fuels"] = np.nan

        # transform-aware covariates
        def _resolve_base(base: str) -> Optional[str]:
            if base in g.columns:
                return base
            wide = [c for c in g.columns if c.startswith(base + "__")]
            if wide:
                g[base] = g[wide].sum(axis=1, min_count=1)
                return base
            tot = base + "_total"
            if tot in g.columns:
                return tot
            return None

        # Default: with FD outcomes, prefer lagged levels for controls
        prefer_lag_if_fd = bool(getattr(cfg, "use_lag_levels_in_diff", True))

        transform_hint = {
            x: ("diff" if x.startswith("d_") else ("lag1" if x.startswith("L1_") else None))
            for x in requested_raw
        }
        base_for = {x: _base_name(x) for x in requested_raw}

        resolved_by_base: Dict[str, Optional[str]] = {}
        for b in set(base_for.values()):
            resolved_by_base[b] = _resolve_base(b)

        covar_cols_used: List[str] = []
        seen = set()

        def _append_if_valid(colname: str) -> None:
            if colname in g.columns and g[colname].notna().sum() >= 3 and float(g[colname].var(skipna=True)) > 0:
                if colname not in seen:
                    covar_cols_used.append(colname)
                    seen.add(colname)

        # If user specified covariates, honor them (with transformations)
        if requested_raw:
            for req in requested_raw:
                base = base_for[req]
                col0 = resolved_by_base.get(base)
                if col0 is None:
                    continue
                hint = transform_hint[req]
                if hint == "lag1":
                    cname = f"L1_{base}"
                    g[cname] = g.groupby("unit_id", sort=False)[col0].shift(1)
                elif hint == "diff":
                    cname = f"d_{base}"
                    g[cname] = g.groupby("unit_id", sort=False)[col0].diff()
                else:
                    if cfg.differenced and prefer_lag_if_fd:
                        cname = f"L1_{base}"
                        g[cname] = g.groupby("unit_id", sort=False)[col0].shift(1)
                    elif cfg.differenced:
                        cname = f"d_{base}"
                        g[cname] = g.groupby("unit_id", sort=False)[col0].diff()
                    else:
                        cname = col0
                _append_if_valid(cname)
        else:
            # Conservative defaults
            defaults = ["renewable_to_fossil_supply_ratio"]
            if "energy_demand_fossil_fuels" in g.columns:
                defaults.append("energy_demand_fossil_fuels")
            # include nuclear share by default but respect FD rule
            defaults.append("nuclear_share_supply")

            for base in defaults:
                col0 = _resolve_base(base) or base
                if cfg.differenced and prefer_lag_if_fd:
                    cname = f"L1_{base}"
                    # already have explicit lag for nuclear_share_supply, but compute generically
                    g[cname] = g.groupby("unit_id", sort=False)[col0].shift(1)
                elif cfg.differenced:
                    cname = f"d_{base}"
                    g[cname] = g.groupby("unit_id", sort=False)[col0].diff()
                else:
                    cname = col0
                _append_if_valid(cname)

        # leakage-safe scaling (fit on pre rows only)
        covs_scaler_path = None
        if covar_cols_used:
            fit_on = getattr(cfg, "fit_on", "never_or_notyet")
            pre_mask = (
                (g["treated_ever"] == 0)
                if (fit_on == "controls_only")
                else ((g["treated_ever"] == 0) | (g[cfg.year_col] < g["g"]))
            )
            X_pre = g.loc[pre_mask, covar_cols_used].copy()
            pre_medians = {c: X_pre[c].median(skipna=True) for c in covar_cols_used}
            for c in covar_cols_used:
                g[c] = g[c].fillna(pre_medians[c])

            scaler = StandardScaler()
            scaler.fit(g.loc[pre_mask, covar_cols_used].values)
            g[covar_cols_used] = scaler.transform(g[covar_cols_used].values)

            if getattr(cfg, "artifact_dir", None):
                Path(cfg.artifact_dir).mkdir(parents=True, exist_ok=True)
                import joblib as _joblib
                covs_scaler_path = os.path.join(cfg.artifact_dir, "covariates_scaler_prefit.joblib")
                _joblib.dump({"scaler": scaler, "columns": covar_cols_used, "pre_medians": pre_medians},
                             covs_scaler_path)
                print(f"[prepare] Saved covariates scaler -> {covs_scaler_path}")

        # ---------- (7) Trim units by support (differencing-aware) ----------
        effective_min_pre = max(1, cfg.min_pre - (1 if cfg.differenced else 0))
        effective_min_post = cfg.min_post
        keep_units: List[str] = []
        for uid, dfu in g.groupby("unit_id", sort=False):
            ever = int(dfu["treated_ever"].iloc[0])
            if ever:
                pre = dfu[dfu[cfg.year_col] < dfu["g"].iloc[0]]
                post = dfu[dfu["post"] == 1]
                if len(pre) >= effective_min_pre and len(post) >= effective_min_post:
                    keep_units.append(uid)
            else:
                if dfu[cfg.year_col].nunique() >= (effective_min_pre + effective_min_post):
                    keep_units.append(uid)
        dropped_units = sorted(set(g["unit_id"].unique()) - set(keep_units))
        if dropped_units:
            print(f"[prepare] Dropped {len(dropped_units)} units with insufficient pre/post support.")
        g = g[g["unit_id"].isin(keep_units)].copy()

        # ---------- (8) Bin diagnostics after trimming (no re-binning) ----------
        for b in sorted(g[g["dose_bin"].notna()]["dose_bin"].unique()):
            doses = g.loc[(g["dose_bin"] == b) & (g["post"] == 1), "dose_level"]
            print(f"  - {b}: dose range=[{doses.min():.1f}, {doses.max():.1f}], "
                f"CV={doses.std() / doses.mean():.2f}")
            

        # Add to preparation.py
        # Test if covariate is affected by treatment (bad control check)
        treated = g[g["treated_ever"] == 1]
        pre = treated[treated["post"] == 0].groupby("unit_id")["nuclear_share_supply"].mean()
        post = treated[treated["post"] == 1].groupby("unit_id")["nuclear_share_supply"].mean()
        delta = post - pre
        if abs(delta.mean()) > 0.05:  # 5 percentage point threshold
            print(f"[WARNING] nuclear_share_supply changes {delta.mean():.3f} post-treatment (potential bad control)")


        # ---------- (9) Stats & info ----------
        units = g["unit_id"].nunique()
        ever = int(g.groupby("unit_id")["treated_ever"].max().sum())
        treated_rows = int((g["post"] == 1).sum())
        print("=== PANEL STATS ===")
        print(f"Units: {units} | Ever-treated: {ever} | Obs: {len(g)} | Post rows: {treated_rows} ({treated_rows/len(g):.1%})")

        cohorts = g.loc[g["treated_ever"] == 1, ["unit_id", "g"]].drop_duplicates()
        cohort_counts = cohorts["g"].value_counts().sort_index()

        self.covar_cols_used = covar_cols_used
        self.info = {
            "unit_level": unit_level,
            "n_sd_wide_cols": len(sd_wide_names),
            "pca": {
                "vars_used": pca_info.get("vars_used"),
                "pc1_explained": pca_info.get("pc1_explained"),
                "scaler_path": pca_info.get("scaler_path"),
                "model_path": pca_info.get("model_path"),
            },
            "covs_scaler_path": covs_scaler_path,
            "dropped_units": dropped_units,
            "covariates_used": covar_cols_used,
            "emissions_outcome_mode_used": used_outcome_mode,
            "outcome_var": cfg.outcome_col,
            "differenced": cfg.differenced,
            "use_lag_levels_in_diff": bool(getattr(cfg, "use_lag_levels_in_diff", True)),
            "outcome_name": self.outcome_name,
            "dose_bin_support": dose_bin_support,
            "dose_bin_method": (dose_bin_support or {}).get("method"),
            "dose_bin_edges": (dose_bin_support or {}).get("edges"),
            "pre": int(getattr(cfg, "pre", 0)),
            "post": int(getattr(cfg, "post", 0)),
            "pre": int(getattr(cfg, "pre", 0)),
            "post": int(getattr(cfg, "post", 0)),
            "cohort_counts": cohort_counts.to_dict(),
            "tau": cfg.treat_threshold,
            "n_bins": (int(getattr(cfg, "n_bins", 0)) or None),
        }

        # ---------- (10) Final projection ----------
        base_keep = [
            country_col, cfg.year_col, "unit_id",
            "dose_level", "treated_now", "treated_ever", "g", "post", "event_time"
        ]
        if unit_level == "sector":
            base_keep.insert(1, ccus_col)

        panel_cols = base_keep + [self.outcome_name] + covar_cols_used
        # always keep dose_bin if present
        if "dose_bin" in g.columns:
            panel_cols.append("dose_bin")
        # expose nuclear share transforms (even if not selected as controls) for diagnostics
        for extra in ["nuclear_share_supply", "L1_nuclear_share_supply", "d_nuclear_share_supply"]:
            if extra not in panel_cols and extra in g.columns:
                panel_cols.append(extra)

        panel_cols = [c for c in panel_cols if c in g.columns]
        self.panel = g[panel_cols].copy()


# public entry point (backward compatible signature)
def prepare_ccus_panel(cfg: StudyConfig) -> Tuple[pd.DataFrame, List[str], Dict[str, Any], str]:
    pdata = PanelData(cfg)
    return pdata.panel, getattr(pdata, "covar_cols_used", []), pdata.info, pdata.outcome_name
