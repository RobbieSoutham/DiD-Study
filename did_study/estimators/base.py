"""
Base classes for difference‑in‑differences estimators.

This module defines the :class:`BaseEstimator` which stores the study
configuration and provides simple helper methods for printing formulas
and validating regression models.  All estimator classes in the
package should inherit from this base class.
"""

from __future__ import annotations

import numpy as np

from ..helpers.config import StudyConfig


class BaseEstimator:
    """Base class for difference‑in‑differences estimators.

    The base estimator encapsulates common functionality such as
    storing the configuration, printing regression formulas for
    transparency and performing simple model diagnostics.  Concrete
    estimators should subclass this and call ``super().__init__(config)``
    in their own initialiser.

    Parameters
    ----------
    config : :class:`did_study.config.StudyConfig`
        The study configuration describing how the panel was prepared
        and specifying bootstrap and estimation settings.
    """

    def __init__(self, config: StudyConfig) -> None:
        self.config = config

    def print_formula(self, formula: str) -> None:
        """Print the regression formula used in a model.

        Parameters
        ----------
        formula : str
            A patsy style formula defining the regression model.
        """
        print(f"Regression formula: {formula}")

    def validate_model(self, model: object) -> None:
        """Perform simple diagnostic checks on a fitted model.

        This helper reports the number of parameters and how many are
        ``NaN``.  Subclasses may override this method to implement
        more advanced diagnostics if desired.

        Parameters
        ----------
        model : Any
            A fitted model object from StatsModels with a ``params``
            attribute.
        """
        try:
            params = getattr(model, "params", None)
            if params is not None:
                n_par = len(params)
                n_nan = int(np.sum(~np.isfinite(params)))
                print(f"Model has {n_par} parameters; {n_nan} are NaN.")
        except Exception as e:
            print(f"Model validation encountered an error: {e}")
