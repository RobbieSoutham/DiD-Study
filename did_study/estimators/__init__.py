"""Public API for the estimators subpackage.

This module reexports the primary estimator functions and result
containers for convenience.  Users may import these names directly
from :mod:`did_study.estimators`.
"""

from .att import AttResult, estimate_att_o
from .bins import BinAttResult, estimate_binned_att_o
from .event_study import EventStudyResult, event_study

__all__ = [
    "AttResult",
    "estimate_att_o",
    "BinAttResult",
    "estimate_binned_att_o",
    "EventStudyResult",
    "event_study",
]