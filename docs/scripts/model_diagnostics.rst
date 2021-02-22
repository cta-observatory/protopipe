.. _model_diagnostics:

Models diagnostics
==================

.. warning::

  This script constituted basically the benchmarks for the estimator models.
  It is still available, but its contents have been migrated to the benchmark notebooks
  and it is now progressively discontinued.

To get diagnostic plots in order to control the robustness and the performance
of the models you can use the script `model_diagnostic.py`. It takes as arguments
a configuration file:

.. code-block::

    usage: model_diagnostic.py [-h] --config_file CONFIG_FILE [--wave | --tail]

    Make diagnostic plot

    optional arguments:
      -h, --help            show this help message and exit
      --config_file         CONFIG_FILE
      --wave                if set, use wavelet cleaning
      --tail                if set, use tail cleaning, otherwise wavelets

For the energy estimator the diagnostic plots consist in:

* Distribution of the features
* Importance of the features
* Distribution of the ratio of the reconstructed energy over the true energy
  fitted with a gaussian for the subarrays
* Energy resolution and energy bias corresponding to the gaussian parametrisation
  for the subarrays

For a g/h classifier the following diagnostic are provided:

* Distribution of the features
* Importance of the features
* ROC curve (and its variation with energy)
* Output model distribution (and its variation with energy)
