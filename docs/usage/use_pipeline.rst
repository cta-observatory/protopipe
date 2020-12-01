.. _use-pipeline:

Pipeline
========

The following steps describe the basics of protopipe analysis and
estimate of the performance of CTA.

Each of the analysis steps comes with its configuration file stored under
``aux/example_config_files/protopipe``.

1. Energy estimator

* produce a table with gamma-ray image information
  with pipeline utilities (:ref:`data_training`)
* build a model with ``protopipe.mva`` utilities (:ref:`model_building`)

2. Gamma hadron classifier

* produce tables of gamma-rays and hadrons with image information
  with pipeline utilities (:ref:`data_training`)
* build a model with ``protopipe.mva`` utilities (:ref:`model_building`)

3. DL2 production

* produce tables of gamma-rays, hadrons and electrons with event informations
  with pipeline utilities (:ref:`DL2`)

4. Estimate performance of the instrument (:ref:`optimization_cuts_IRFs`)

* find the best cutoff in gammaness/score, to discriminate between signal
  and background, as well as the angular cut to obtain the best sensitivity
  for a given amount of observation time and a given template for the
  source of interest
* compute the instrument response functions, effective area,
  point spread function and energy resolution
* estimate the sensitivity

.. warning::

  *protopipe* is currently optimized for point-source best-sensitivity
