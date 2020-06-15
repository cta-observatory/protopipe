.. _use-pipeline:

Pipeline
========

The following steps describe the basics of protopipe analysis and estimate of the performance of CTA.

.. warning::
  | Even though *protopipe* can now support all prod3b simtel Monte Carlo files,
    it is currently tested using only LSTCam and NectarCam cameras.
  | Note that some generic La Palma files can contain FlashCam cameras.

1. Setup the analysis

* create an analysis parent folder with the auxiliary script ``create_dir_structure.py``
* ``python create_dir_structure.py $PATH $NAME``
* copy and edit the example YAML configuration files in the *config* subfolders

2. Energy estimator

* produce a table with gamma-ray image information with pipeline utilities (:ref:`data_training`)
* build a model with ``protopipe.mva`` utilities (:ref:`model_building`)

3. Gamma hadron classifier

* produce tables of gamma-rays and hadrons with image information with pipeline utilities (:ref:`data_training`)
* build a model with ``protopipe.mva`` utilities (:ref:`model_building`)

4. DL2 production

* produce tables of gamma-rays, hadrons and electrons with event informations with pipeline utilities (:ref:`DL2`)

5. Estimate performance of the instrument (:ref:`optimization_cuts_IRFs`)

* find the best cutoff in gammaness/score, to discriminate between signal
  and background, as well as the angular cut to obtain the best sensitivity
  for a given amount of observation time and a given template for the
  source of interest
* compute the instrument response functions, effective area,
  point spread function and energy resolution
* estimate the sensitivity

.. warning::

  * *protopipe* currently is optimized for point-source best-sensitivity,
  * DL1/DL2 scripts take as input only 1 file at the time.
