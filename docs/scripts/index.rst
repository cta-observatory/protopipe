.. _scripts:

scripts
=======

Introduction
------------

This module contains the scripts that make up the pipeline operational sequence
explained in :ref:`usage`.

Each step of the analysis is explained in a dedicated section below.


Details
-------

.. toctree::
    :maxdepth: 1

    data_training
    model_building
    model_diagnostics
    DL2
    optimization_cuts_IRFs

The tables created in the :ref:`data_training` and :ref:`DL2` steps are written
on disk in HDF5_ format using the PyTables_ Python module.
This choice has been done considering the potential huge
volume of data we will be dealing with in a near future.

.. Reference/API
.. -------------
..
.. .. automodapi:: protopipe.scripts
..     :no-inheritance-diagram:
    .. :include-all-objects:

.. _PyTables: https://www.pytables.org/
.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
.. _gamma astro data format: https://gamma-astro-data-formats.readthedocs.io/en/latest/index.html
