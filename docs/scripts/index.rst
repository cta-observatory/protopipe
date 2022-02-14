.. _scripts:

Scripts (``protopipe.scripts``)
===============================

This module contains the scripts that make up the pipeline operational sequence
explained in :ref:`usage`.

The tables created in the :ref:`data_training` and :ref:`DL2` steps are written
on disk in HDF5_ format using the PyTables_ Python module.
This choice has been done considering the potential huge
volume of data we will be dealing with in a near future.  

The :ref:`model_building` script saves models, parameter values and lists of train/test data
in a pickled format.  

The script performing :ref:`optimization_cuts_IRFs` saves DL3 data in FITS format
following CTA experimental extensions of the `gamma astro data format`_.

.. toctree::
    :maxdepth: 1

    data_training
    model_building
    DL2
    optimization_cuts_IRFs



.. _PyTables: https://www.pytables.org/
.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
.. _gamma astro data format: https://gamma-astro-data-formats.readthedocs.io/en/latest/index.html
