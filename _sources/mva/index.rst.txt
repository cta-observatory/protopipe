.. _mva:

mva
===

Introduction
------------

`protopipe.mva` contains utilities to build models for regression or
classification problems. It is based on machine learning methods available in
scikit-learn_. Internally, the tables are dealt with the Pandas_ Python module.

For each type of camera a regressor/classifier should be trained. For both type of models
an average of the image estimates is later computed to determine a global
output for the event (energy or score/gammaness).

Details
-------

The class `TrainModel` uses a training sample composed of gamma-rays for a
regression model. In addition of a gamma-ray sample, a sample of
protons is also used to build a classifier. The training of a model is done via
the GridSearchCV_ algorithm which allows to find the best hyper-parameters of
the models.

The RegressorDiagnostic and ClassifierDiagnostic classes can be used to generate
several diagnostic plots for regression and classification models, respectively.

Proposals for improvements and/or fixes
---------------------------------------

.. note::

  This section has to be moved to the repository as a set of issues.

* Improve split of training/test data. For now the split of the data is done
  according to the run number, e.g. training data will be N% of the first runs
  (sorted by run numbers) and test data will be the remaining runs. Really easy
  to improve with scikit-learn. But I wanted to keep the information about evt_id and
  the obs_id in order to combine the data and produce diagnostic plot at the
  level of event (not implemented yet), which is more complex that what scikit does.
* Implement event-level diagnostic.
* To train the energy estimator, the Boosted Decision Tree method is hard-coded.
* For the diagnostic, in both case we might want to implement diagnostics
  at the level of events but for this we need to link the event Id with the
  observation Id as well as the image parameters to split and combine the
  model output. It needs some thoughts...

Reference/API
-------------

.. automodapi:: protopipe.mva
    :no-inheritance-diagram:
    :skip: auc, roc_curve

.. _scikit-learn: https://scikit-learn.org/
.. _GridSearchCV: https://scikit-learn.org/stable/modules/grid_search.html
.. _Pandas: https://pandas.pydata.org/
