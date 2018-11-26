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
 * build regression or classification models along with diagnostic plots
 * estimate the best cutoffs which gives the minimal sensitivy
   reachable in a given amount of time
 * produce instrument response functions (IRF), including sensitivity

In order to process a significant amount of events the use of the GRID is rapidly
mandatory. Some utility scripts to submits jobs on the GRID are provided on
GRID.

Documentation
=============

.. toctree::
   :maxdepth: 1

   pipeline/index
   mva/index
   perf/index
   scripts/index
   
What should be improved?
========================
 * Saving calibrated images as a DL1 format (here, only image parameters are saved)
 * The EventPreparer is a bit messy, it should returns containers
 * Features for regression and classification are hard-coded in the two scripts
 `write_dl1.py` and `write_dl2.py`
 * Data format

   
