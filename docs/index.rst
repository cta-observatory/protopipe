============================
A pipeline protopipe for CTA
============================

.. currentmodule:: protopipe

What is protopipe?
==================
`Protopipe` is a pipeline prototype for the `Cherenkov Telescope Array
<https://www.cta-observatory.org/>`_ (CTA) based on the `ctapipe
<https://cta-observatory.github.io/ctapipe/>`_ library.
The package is currently developped and tested at the CEA in the departement
of astrophysics.

The pipeline provides scripts to:
 * Process simtelarray files and write DL1 or DL2 tables
 * Build regression or classification models with diagnostic plots
 * Estimate the best cutoffs which gives the minimal sensitivy
   reachable in a given amount of time
 * Produce instrument response functions (IRF), including sensitivity

In order to process a significant amount of events the use of the GRID is rapidly
mandatory. Some utility scripts to submits jobs on the GRID are provided on
the `GRID repository <https://drf-gitlab.cea.fr/CTA-Irfu/grid>`_.

Installation
============
It is recommanded to build a new environment with conda.
You can follow the instructions on the `ctapipe installation`_ page.
In order to use protopipe the following modules are required:
 * the `ctapipe`_ library
 * the `gammapy`_ library
 * the `pywi-cta`_ module

In order to install protopipe , try `python setup.py` in your conda environment.

How to?
=======
For this pipeline prototype, in order to build an analysis to estimate
the performance of the instruments, a user will follows the following steps:
 1. Energy estimator
  * produce a table with gamma-ray image information with pipeline utilities (:ref:`pipeline`)
  * build a model with mva utilities (:ref:`mva`)
 2. Gamma hadron classifier
  * produce tables of gamma-rays and hadrons with image informations with pipeline utilities (:ref:`pipeline`)
  * build a model with mva utilities (:ref:`mva`)
 3. DL2 production
  * produce tables of gamma-rays, hadrons and electrons with event informations with pipeline utilities (:ref:`pipeline`)
 4. Estimate performance of the instrument
  * find the best cutoff in gammaness/score, to discriminate between signal
    and background, as well as the angular cut to obtain the best sensitivity
    for a given amount of observation time and a given template for the
    source of interest (:ref:`perf`)
  * compute the instrument response functions, effective area,
    point spread function and energy resolution (:ref:`perf`)
  * estimate the sensitivity (:ref:`perf`)

Documentation
=============

.. toctree::
   :maxdepth: 1

   pipeline/index
   mva/index
   perf/index
   scripts/index

.. _ctapipe installation: https://cta-observatory.github.io/ctapipe/getting_started/index.html#step-4-set-up-your-package-environment
.. _ctapipe: https://cta-observatory.github.io/ctapipe/
.. _gammapy: https://gammapy.org/
.. _pywi: http://www.pywi.org/
.. _pywi-cta: http://cta.pywi.org/
