.. _benchmarks:

Benchmarks (``protopipe.benchmarks``)
=====================================

The ``protopipe.benchmarks`` module is composed of 3 parts:

- ``notebooks``, a folder containing benchmarking Jupyter notebooks divided by analysis stage,
- ``operations``, a sub-module containing functions to perform many different operations on data related to benchmarking
- ``plot``, a sub-module containing plotting functions
- ``utils``, a sub-module containing utility functions for the notebooks
- ``book_template``, a folder containing the Jupyter Book template for a CTA analysis

.. note::
    Much of what is contained in the sub-modules is the product of a long refactoring process
    of old material from the notebooks.  
    Many things can be improved or imported by *ctaplot*/*cta-benchmarks* and *ctapipe* 
    as the refactoring of the pipeline takes progress.  
    Also, not all notebooks are exatcly the same in terms of global options,
    a notebook template will be added.

.. note::
    All benchmarks will be launched by means of a new ``protopipe-BENCHMARK`` script.
    This will become the recommended method, as it will integrates with the rest of the analysis interface.

API reference
-------------

.. automodapi:: protopipe.benchmarks
