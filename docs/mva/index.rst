.. _mva:

=============================
MultiVariate Analysis (`mva`)
=============================

Introduction
============

`protopipe.mva` contains utilities to build model, for regression or classification
problems, based on machine learning methods available in `scikit-learn <https://scikit-learn.org/>`_. For each type of camera a regressor/classifier is trained and for each case an average of the estimates is done to determine an event output.

For now a Boosted Decision Tree regressor is trained to build an energy estimator,
with. Trees

ToDo
====
* Improve split of training/test data. For now the split of the data is done
  according to the run number, e.g. training data will be N% of the first runs
  (sorted by run numbers) and test data will be the remaining runs

Warnings
========
* process simtelarray files and write DL1 or DL2 tables
  
Reference/API
=============

.. automodapi:: protopipe.mva
   :no-inheritance-diagram:
