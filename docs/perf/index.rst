.. _perf:

perf
====

Introduction
------------

The DL2-to-DL3 step of the *protopipe* pipeline is now based on the 
`pyirf <https://cta-observatory.github.io/pyirf/>`_ library.

The ``protopipe.perf`` module contains utility tools used to interface with this
library, mainly to translate the DL2 data in the internal nomenclature used by
*pyirf* to produce the DL3 file.

In general this step takes care of

* handling the determination of the best-cutoffs to separate gamma and
  the background (protons + electrons),
* estimating the sensitivity from the optimized cuts,
* producing the instrument response functions (IRFs).


Provided that a performance script makes used of 
``protopipe.perf.utils.read_DL2_pyirf``, *protopipe* supports the addition and 
testing of multiple scripts based on pyirf.

.. note::
  Some functions are discontinued or will be with the next release
  and they will be removed.

The current approach is based on the EventDisplay historical pipeline for which
the following main points apply,

- minimum telescope multiplicity cut set to 3 (for 30' and 100'' exposures)
  and 4 (for 5h and 50h exposures),
- maximum signal efficiency set to 80%,
- optimisation cuts are performed following these steps,
  - applying a global cut of 40% signal efficiency,
  - energy-bin-wise 68% containment angular cuts optimization 
  - energy-bin-wise optimisation of gamma/hadron separation cut for best sensitivity

For more details on the specific API and benchmarks, please check *pyirf*'s documentation.

Reference/API
-------------

.. automodapi:: protopipe.perf
    :no-inheritance-diagram:
    :include-all-objects:
    :skip: calculate_bin_indices, inter_quantile_distance

.. _Gammapy: https://gammapy.org/
