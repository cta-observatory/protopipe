.. _requirements:

************
Requirements
************

Basic
=====

- ``Python >= 3.8``
- ``pip``

*protopipe* can be installed using only ``pip``, so any virtual environment solution
providing it should suffice.
Though, if you want to use *protopipe* on the *DIRAC* grid, an *Anaconda*-based virtual
environment is the only currently supported solution.

.. note::
  In case you want to use Anaconda it is recommended to use
  `mamba <https://github.com/mamba-org/mamba#readme>`__
  for improved speed during the creation of the virtual environment.

Optional
========

In order to **open data files** stored as HDF5 tables it is suggested to
integrate the final virtual environment with the ``vitables`` package.

If you plan to compare *protopipe* results against older results based on **CERN/ROOT**
from the historical pipelines (*CTAMARS* and/or *EventDisplay*)
you will need to install also the ``uproot`` package.

**Benchmarking** can make use of the ``seaborn`` package, 
if enabled from the associated configuration file (for more details see :ref:`benchmark_script`).