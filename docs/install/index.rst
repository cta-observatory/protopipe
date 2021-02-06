.. _install:

Installation
============

The only requirement is an Anaconda (or Miniconda) installation which supports
Python 3.

There are two different ways to install `protopipe`,

* if you just want to use it as it is (:ref:`install-release`),
* or if you also want to develop it (:ref:`install-development`).

.. note::
  For both types of installation the use of `mamba <https://github.com/mamba-org/mamba#readme>`__
  might provide improved speed during the creation of the `conda` environment.

In both cases, if you want to perform a full analysis, you will need some
computational power in order to produce enough
data files for model and performance estimation.
This can be accomplished through the use of a GRID environment.

After installing `protopipe`,

* install the code necessary to interface it with the grid (:ref:`install-grid`),
* use protopipe on the grid (:ref:`use-grid`).

.. toctree::
    :hidden:
    :maxdepth: 1

    release
    development
    grid
