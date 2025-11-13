# did_study/robustness/wcb.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .r_interface import (
    make_feols_in_r,
    boottest_fixest,
    mboottest_fixest,
)


@dataclass
class FitSpec:
    outcome: str
    regressors: Sequence[str]
    fe: Sequence[str]
    cluster: Sequence[str]


@dataclass
class TestSpec:
    # Scalar test: test this single coefficient == 0
    param: Optional[str] = None
    # Joint test: all coefficients in this list == 0 (uses mboottest)
    joint_zero: Optional[Sequence[str]] = None


class WildClusterBootstrap:
    """
    Thin wrapper around fwildclusterboot (boottest / mboottest).

    - Scalar tests (param != None, joint_zero is empty) use fwildclusterboot::boottest.
    - Joint tests (joint_zero not empty) use fwildclusterboot::mboottest.

    Fits a single fixest::feols model in R via make_feols_in_r and reuses it
    for all bootstrap tests on the same design.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fit_spec: FitSpec,
        B: int = 9999,
        weights: str = "rademacher",
        seed: Optional[int] = 123,
        impose_null: bool = True,
    ) -> None:
        self.df = df
        self.fit_spec = fit_spec
        self.B = int(B)
        self.weights = (weights or "rademacher").lower()
        self.seed = seed
        self.impose_null = bool(impose_null)

        # Fit fixest model in R and retain the sanitized dataframe
        self._r_fit, self._sub_df, self.formula = make_feols_in_r(
            df=df,
            outcome=fit_spec.outcome,
            regressors=fit_spec.regressors,
            fe=fit_spec.fe,
        )

    def _print_header(self, test_spec: TestSpec) -> None:
        """Debug print of the current WCB call."""
        kind = "boottest_fixest"
        if test_spec.joint_zero:
            kind = "mboottest_fixest"

        print("=" * 72)
        print(f"[WCB CALL] -> {kind}")
        print("=" * 72)
        print("Parameters:")
        print(f"  - outcome: {self.fit_spec.outcome}")
        print(f"  - regressors: {list(self.fit_spec.regressors)}")
        print(f"  - fe: {list(self.fit_spec.fe)}")
        print(f"  - cluster: {list(self.fit_spec.cluster)}")
        print(f"  - param: {test_spec.param}")
        print(f"  - joint_zero: {test_spec.joint_zero}")
        print(f"  - B: {self.B}")
        print(f"  - type/weights: {self.weights}")
        print(f"  - impose_null: {self.impose_null}")
        print(f"  - seed: {self.seed}")
        print(f"  - Formula: {self.formula}\n")

    def pvalue(self, test_spec: TestSpec) -> float:
        """
        Return a single wild-cluster bootstrap p-value.

        - If test_spec.joint_zero is non-empty, perform a joint test with
          fwildclusterboot::mboottest via mboottest_fixest (returns float).

        - Else, if test_spec.param is not None, perform a scalar test with
          fwildclusterboot::boottest via boottest_fixest (returns an R object
          from which we extract the p-value).

        If the R call fails, we print the reason and return NaN.
        """
        self._print_header(test_spec)

        #try:
        # 1) Joint test: use mboottest_fixest -> returns float p-value
        if test_spec.joint_zero:
            pval = mboottest_fixest(
                df=self._sub_df,
                fit=self._r_fit,
                joint_params=list(test_spec.joint_zero),
                B=self.B,
                impose_null=self.impose_null,
                clustid=self.fit_spec.cluster,
                type_=self.weights,
            )
            return float(pval)

        # 2) Scalar test: use boottest_fixest -> returns R object
        if test_spec.param is not None:
            res = boottest_fixest(
                fit=self._r_fit,
                df=self._sub_df,
                param=str(test_spec.param),
                B=self.B,
                clustid=self.fit_spec.cluster,
                type_=self.weights,
                impose_null=self.impose_null,
                seed=self.seed,
            )

            # Extract p-value from fwildclusterboot::boottest object
            try:
                if "p_val" in res.names:
                    return float(np.array(res.rx2("p_val"))[0])
                elif "p.value" in res.names:
                    return float(np.array(res.rx2("p.value"))[0])
                else:
                    # Fallback: treat as scalar numeric
                    return float(res[0])
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"Could not extract p-value from boottest result: {e}"
                ) from e

        """except Exception as e:
            print(f"[WCB] R test failed; Reason: {e}")"""

        # If anything goes wrong, return NaN so callers can handle gracefully
        return float("nan")


def make_wcb_runner(
    df: pd.DataFrame,
    outcome: str,
    regressors: Sequence[str],
    fe: Sequence[str],
    cluster: Sequence[str],
    B: int = 9999,
    weights: str = "rademacher",
    impose_null: bool = True,
    seed: Optional[int] = 123,
) -> WildClusterBootstrap:
    """Convenience constructor used by the estimators (att, bins, event_study)."""
    fit_spec = FitSpec(
        outcome=outcome,
        regressors=list(regressors),
        fe=list(fe),
        cluster=list(cluster),
    )
    return WildClusterBootstrap(
        df=df,
        fit_spec=fit_spec,
        B=B,
        weights=weights,
        seed=seed,
        impose_null=impose_null,
    )
