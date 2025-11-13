"""
Robustness utilities for difference-in-differences estimators.

This subpackage groups together modules that provide diagnostic and
robustness checks for DiD analyses.  The contents are organised to
facilitate extensibility and reuse.  Currently there are three
primary components:

* :mod:`did_study.robustness.stats` contains general statistics
  routines including pre-trend tests and minimum detectable effect
  calculations.
* :mod:`did_study.robustness.wcb` wraps the R package
  ``fwildclusterboot`` via ``rpy2`` to compute wild cluster
  bootstrap p-values for single parameters and joint hypotheses.
* :mod:`did_study.robustness.honest_did` implements Rambachan-Roth
  sensitivity bounds ("Honest DiD") for assessing robustness to
  violations of the parallel trends assumption.

The goal of collecting these functions in a ``robustness`` folder is
to keep the main estimator modules lightweight while enabling easy
extension with additional robustness checks in the future.  Users may
import any of these functions directly from this package; the most
common helpers are reexported below for convenience.

Examples
--------
Compute a wild cluster bootstrap p-value for a single coefficient::

    from did_study.robustness import wcb_att_pvalue_r
    p = wcb_att_pvalue_r(df, outcome='y', regressors=['treat'], fe=['Year'],
                         cluster='unit_id', param='treat', B=9999,
                         weights='rademacher')

Compute a joint pre-trend test::

    from did_study.robustness import joint_pretest_zero
    result = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
    pre_names = ['ES_tm3','ES_tm2']
    joint = joint_pretest_zero(result, pre_names)
    print(joint['p_value'])

Compute Rambachan-Roth HonestDiD bounds::

    from did_study.robustness import honest_did_bounds
    bh = es_result.coefs['beta'].to_numpy()
    bounds = honest_did_bounds(bh, num_pre_periods=4, num_post_periods=10,
                               M=0.5, bound_type='relative')
    print(bounds['lower'], bounds['upper'])

"""