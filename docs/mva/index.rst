.. _mva:

=============================
MultiVariate Analysis (`mva`)
=============================

Introduction
============

`protopipe.mva` contains utilities to build models for regression or
classification problems. It is based on machine learning methods available in
scikit-learn_. Internally, the tables are dealt with the Pandas_ Python module.

For each type of camera a regressor/classifier bould be trained For both type of models
an average of the image estimates is later computed to determine a global
output for the event (energy or score/gammaness).

The class `TrainModel` uses a training sample composed of gamma-rays for a
regression model. In addition of a gamma-ray sample, a sample of
protons is also used to build a classifier. The training of a model is done via
the GridSearchCV_ algorithm which allows to find the best hyper-parameters of
the models.

The RegressorDiagnostic and ClassifierDiagnostic classes should be used to generate
several diagnostic plots for regression and classification models, respectively.

How to build an energy estimator
================================
To build a regressor you need a table with at least the MC energy (the target)
and some event caracteristics (the features) to reconstruct the energy. The
script called `protopipe.scripts.build_model.py` takes as arguments a configuration
file:

.. code-block:: bash

    (cta_pipeline) [julien@TheVerse] scripts $ ./build_model.py --help
    usage: build_model.py [-h] --config_file CONFIG_FILE [--max_events MAX_EVENTS]
                      [--wave | --tail]

Here is an example of a configuration file with some comments to build a model:

.. code-block:: yaml

    General:
     model_type: 'regressor'  # regressor or classifier
     data_dir: 'absolute_data_path'  # Directory with the data
     data_file: 'dl1_{}_gamma_merged.h5'  # Name of the data file ({} will be completed with the mode (tail,wave))
     outdir: 'absolute_output_path'  # Output directory
     cam_id_list: ['LSTCam', 'NectarCam']  # List of camera
     table_name_template: 'feature_events_'  # Template name of table in DF5 (will be completed with cam_ids)

    Split:
     train_fraction: 0.8  # Fraction of events use to train the regressor

    Method:
     name: 'AdaBoostRegressor'  # Scikit-learn model name
     target_name: 'mc_energy'  # Name of the regression target
     tuned_parameters:  # List of parameters to optimise the hyperparameters
      learning_rate: [0.1, 0.2, 0.3]
      n_estimators: [100, 200]
      base_estimator__max_depth: [null]  # null is equivalent to None
      base_estimator__min_samples_split: [2]
     scoring: 'explained_variance'  # Metrics to choose the best regressor
     cv: 2  # k in k-cross-validation

    FeatureList:  # List of feature to build the energy regressor
     - 'log10_charge'
     - 'log10_impact'
     - 'width'
     - 'length'
     - 'h_max'

    SigFiducialCuts:  # Fidicual cuts that will be applied on data
     - 'xi <= 0.5'

    Diagnostic:  #  For diagnostic plots
     # Energy binning (used for reco and true energy)
     energy:
      nbins: 10
      min: 0.01
      max: 100

Explain the principle (charge vs impact), RF/BDT, etc.

How to build a g/h classifier
=============================


What could be improved?
=======================
* Improve split of training/test data. For now the split of the data is done
  according to the run number, e.g. training data will be N% of the first runs
  (sorted by run numbers) and test data will be the remaining runs


Reference/API
=============

.. automodapi:: protopipe.mva
   :no-inheritance-diagram:

.. _scikit-learn: https://scikit-learn.org/
.. _GridSearchCV: https://scikit-learn.org/stable/modules/grid_search.html
.. _Pandas: https://pandas.pydata.org/
