from __future__ import annotations

from dataclasses import dataclass

from ..helpers.config import StudyConfig


@dataclass
class BaseEstimator:
    """
    Very small common base class for the estimators in this package.

    At the moment it only stores the :class:`StudyConfig` object and
    exposes a tiny helper for logging formulas, but it can easily be
    extended in the future if shared behaviour is needed.
    """

    config: StudyConfig

    def _log(self, message: str) -> None:
        print(f"[ESTIMATOR] {message}")

    def _log_formula(self, formula: str, extra: str = "") -> None:
        msg = f"Formula: {formula}"
        if extra:
            msg += f" ({extra})"
        self._log(msg)
