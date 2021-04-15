.. _mva:

mva
===

Introduction
------------

`protopipe.mva` contains utilities to build models for regression or
classification problems. It is based on machine learning methods available in
scikit-learn_. Internally, the tables are dealt with the Pandas_ Python module.

For each type of camera a regressor/classifier should be trained.
For both type of models an average of the image estimates is later computed to
determine a global output for the event (energy or score/gammaness).

Details
-------

Data is split in train and test subsamples by single telescope images.

The class ```TrainModel``` uses a training sample composed of gamma-rays for a
regression model. In addition of a gamma-ray sample, a sample of
protons is also used to build a classifier.

The training of a model can be done also via the GridSearchCV_ algorithm which 
allows to find the best hyper-parameters of the models.

Supported models:

- ``sklearn.ensemble.RandomForestClassifier``
- ``sklearn.ensemble.RandomForestRegressor``
- ``sklearn.ensemble.AdaBoostRegressor``

For details about the generation of each model type, please refer to 
:ref:`model_building`.

Reference/API
-------------

.. automodapi:: protopipe.mva
    :no-inheritance-diagram:
    :skip: auc, roc_curve, train_test_split

.. _scikit-learn: https://scikit-learn.org/
.. _GridSearchCV: https://scikit-learn.org/stable/modules/grid_search.html
.. _Pandas: https://pandas.pydata.org/
