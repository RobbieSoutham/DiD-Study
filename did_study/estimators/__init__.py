"""Public API for the estimators subpackage.

This module reexports the primary estimator functions and result
containers for convenience.  Users may import these names directly
from :mod:`did_study.estimators`.

Import directly from submodules to avoid circular imports, e.g.::

    from did_study.estimators.att_contdid import ContdidAttResult
"""

__all__ = [
    "AttResult",
    "estimate_att_o",
    "ContinuousAttResult",
    "estimate_att_o_continuous",
    "ContdidAttResult",
    "estimate_att_o_contdid",
    "BinAttResult",
    "estimate_binned_att_o",
    "EventStudyResult",
    "event_study",
]


def __getattr__(name: str):
    """Lazy imports to avoid circular import issues at package init time."""
    _map = {
        "AttResult": (".att", "AttResult"),
        "estimate_att_o": (".att", "estimate_att_o"),
        "ContinuousAttResult": (".att_continuous", "ContinuousAttResult"),
        "estimate_att_o_continuous": (".att_continuous", "estimate_att_o_continuous"),
        "ContdidAttResult": (".att_contdid", "ContdidAttResult"),
        "estimate_att_o_contdid": (".att_contdid", "estimate_att_o_contdid"),
        "BinAttResult": (".bins", "BinAttResult"),
        "estimate_binned_att_o": (".bins", "estimate_binned_att_o"),
        "EventStudyResult": (".event_study", "EventStudyResult"),
        "event_study": (".event_study", "event_study"),
    }
    if name in _map:
        import importlib
        mod_name, attr = _map[name]
        mod = importlib.import_module(mod_name, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
