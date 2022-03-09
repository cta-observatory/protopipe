.. _reference:

API Reference
=============

This page gives an overview of the public API of *protopipe*.

*protopipe* is composed of the following modules:

- ``scripts``, contains high-level scripts (comparable to ctapipe's tools),
- ``pipeline``, contains the low-level API regarding the event loop, calibration, image cleaning and shower geometry reconstruction,
- ``mva``, is a module for multi-variate analysis containing classes and functions for building machine learning models,
- ``perf``, contains low-level functions used to produce the final performance,
- ``benchmarks``, contains benchmarking notebooks with their utility, computing and plotting functions.

During installation, the package is integrated with auxiliary data containing the YAML configuration files required to launch the scripts.


.. toctree::
    :maxdepth: 1

    scripts
    pipeline
    mva
    perf
    benchmarks
    dirac