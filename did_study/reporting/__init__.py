"""Reporting utilities for the ``did_study`` package.

This subpackage collects functions for summarising and plotting the
results of a DiD study.  The :mod:`did_study.reporting.summary`
module contains a single function, :func:`print_study_summary`,
which prints a concise textual summary of the analysis.  The
functions in :mod:`did_study.reporting.plotting` produce
matplotlib figures for pooled treatment effects, dose‑bin effects
and event study coefficients.

Users may import these functions directly from this subpackage.  For
example::

    from did_study.reporting import print_study_summary, plot_event_study

"""