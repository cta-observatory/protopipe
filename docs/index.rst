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
 * process simtelarray files and write DL1 or DL2 tables
 * build regression or classification along with diagnostic plots
 * estimate the best cutoffs which gives the minimal sensitivy
   reachable in a given amount of time
 * produce instrument response functions (IRF), including sensitivity

In order to process a significant amount of events the use of the GRID is rapidly
mandatory. Some utility scripts to submits jobs on the GRID are provided on
the `GRID repository <https://drf-gitlab.cea.fr/CTA-Irfu/grid>`_.

How to?
=======
For this pipeline prototype, in order to buil a configuration to estimate
the performance of the instruments, a user will follows the following steps:
 * produce a table with gamma-ray image information with pipeline utilities (LINK)
 * build an energy estimator with those gamma-rays image
 * produce tables of gamma-rays and hadrons with image informations
 * build a g/h classifier with those tables
 * produce tables of gamma, electrons and hadrons with direction, energy and
   score/gammaness information
 * find the best cutoff in gammaness/score as well as the angular cut to get
   the best sensitivity for a given amount of observation time and a given
   template for the source of interest
 * estimate the response of the instrument (effective area,
   point spread function, energy resolution, sensitivity)

Documentation
=============

.. toctree::
   :maxdepth: 1

   pipeline/index
   mva/index
   perf/index
   scripts/index
