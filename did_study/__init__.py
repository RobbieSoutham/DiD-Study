"""
The :mod:`did_study` package provides a high-level implementation of modern
difference-in-differences (DiD) methods for staggered treatment adoption.  In
particular, it implements the Average Treatment on the Treated (ATT) summary
parameters proposed by Callaway and Sant'Anna and supports bootstrap based
inference using the wild cluster bootstrap.  The package is built with a
modular, object-oriented design to make it easy to conduct entire research
pipelines from a raw panel to publishable figures.

The package exposes three core classes:

``PanelData``
    Responsible for constructing a clean panel suitable for DiD analysis.  This
    class takes the raw data along with a configuration and produces a
    transformation with a unit identifier, event time variables, difference
    transformed outcomes and a set of covariates.  It also checks minimum
    pre/post support and optionally bin doses.  See
    :class:`did_study.panel.PanelData` for details.

``DidEstimator``
    Implements estimators for pooled ATT, dose-bin specific ATT, and event
    studies with two-way fixed effects.  It also provides functions for
    computing the Minimum Detectable Effect (MDE) and running pretrend tests.
    See :class:`did_study.inference.DidEstimator`.

``DidStudy``
    A high-level orchestrator that wires together panel preparation and
    estimation.  Users instantiate this class with a configuration and call
    :meth:`did_study.study.DidStudy.run` to obtain a dictionary of results and
    figures.  See :class:`did_study.study.DidStudy`.

References
----------
The implementation follows recommendations from several strands of the
literature.  

* Callaway and Sant'Anna describe how to summarize group-time treatment
  effects into a single average treatment on the treated using simple and
  weighted averages[567999609739277^L700-L731].  The package implements
  a collapsed regression to estimate this ``ATT^o`` parameter.  

* Cameron, Gelbach and Miller show that cluster-robust methods can
  over-reject when the number of clusters is small and propose bootstrap
  procedures to improve inference[287988434127442^L33-L47].  We employ the
  wild cluster bootstrap as implemented in the `fwildclusterboot` R package
  to obtain p-values.  

* Simulation studies suggest that wild bootstrap tests based on the
  Rademacher distribution perform well in general, but when the number of
  clusters is below roughly ten, the Webb six-point distribution may perform
  better[359649992076924^L828-L836].  The package therefore chooses between
  ``rademacher`` and ``webb`` weights automatically based on the number of
  clusters.

* Roth (2022) warns that conventional pre-trend tests can have low power and
  conditioning on a passed pre-trend test may increase bias[60792082650145^L16-L47].
  We provide pre-trend assessments primarily for reporting rather than
  conditioning the analysis.

"""
