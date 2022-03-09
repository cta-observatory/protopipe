.. _use-pipeline:

Pipeline
========

The following steps describe the basics of the default analysis with *protopipe* and
estimate of the performance of CTA.

.. note::

  More analysis workflows are possible, but this documentation refers to the default,
  which is currently the only supported one.

Each of the analysis steps uses a configuration file.
Example configuration files are stored under `protopipe/aux/example_config_files`.

1. Build an energy model

* produce tables containing reconstructed image and shower geometry information from gamma rays (:ref:`data_training`)
* build the model (:ref:`model_building`)

2. Build a particle classification model

* produce tables containing reconstructed image and shower geometry information from gamma rays and protons (:ref:`data_training`)
* build the model (:ref:`model_building`)

3. DL2 production

* produce tables containing reconstructed shower geometry, estimated energy and particle type
  for gamma-rays, protons and electrons (:ref:`DL2`)

4. Estimate the final performance (:ref:`optimization_cuts_IRFs`)

* find the best cutoff in gammaness/score, to discriminate between signal
  and background, as well as the angular cut to obtain the best sensitivity
  for a given amount of observation time and a given template for the
  source of interest
* estimate the sensitivity
* compute the instrument response functions, effective area,
  point spread function and energy resolution

.. warning::

  *protopipe* is currently optimized only for point-source best-sensitivity.
