.. _mva:

===
mva
===

Introduction
============

`protopipe.mva` contains utilities to build models for regression or
classification problems. It is based on machine learning methods available in
scikit-learn_. Internally, the tables are dealt with the Pandas_ Python module.

For each type of camera a regressor/classifier should be trained. For both type of models
an average of the image estimates is later computed to determine a global
output for the event (energy or score/gammaness).

The class `TrainModel` uses a training sample composed of gamma-rays for a
regression model. In addition of a gamma-ray sample, a sample of
protons is also used to build a classifier. The training of a model is done via
the GridSearchCV_ algorithm which allows to find the best hyper-parameters of
the models.

The RegressorDiagnostic and ClassifierDiagnostic classes can be used to generate
several diagnostic plots for regression and classification models, respectively.

To know more about how it is done for EvtDisplay and MARS analyses in CTA
please read the IRF document available
`here <https://forge.in2p3.fr/projects/cta_analysis-and-simulations/wiki/Prod3b_based_instrument_response_functions>`_.


How to build models
===================
For both cases, building a regressor or a classifier, the script
`protopipe.scripts.build_model.py` is used.

Energy estimator
----------------
To build a regressor you need a table with at least the MC energy (the target)
and some event characteristics (the features) to reconstruct the energy. The
script called `protopipe.scripts.build_model.py` takes as arguments a configuration
file:

.. code-block:: bash

    >$ ./build_model.py --help
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
     tuned_parameters:  # List of hyperparameters to be optimized
      learning_rate: [0.1, 0.2, 0.3]
      n_estimators: [100, 200]
      base_estimator__max_depth: [null]  # null is equivalent to None
      base_estimator__min_samples_split: [2]  # minimal number to split a node
      base_estimator__min_samples_leaf: [10]  # minimal number of events to form an external node
     scoring: 'explained_variance'  # Metrics to choose the best regressor
     cv: 2  # k in k-cross-validation

    FeatureList:  # List of feature to build the energy regressor
     - 'log10_charge'
     - 'log10_impact'
     - 'width'
     - 'length'
     - 'h_max'

    SigFiducialCuts:  # Fiducial cuts that will be applied on data
     - 'xi <= 0.5'

    Diagnostic:  #  For diagnostic plots
     energy:  # Energy binning (used for reco and true energy)
      nbins: 10
      min: 0.01
      max: 100

To estimate the energy of a gamma-ray, one want to use the relation between
the charge measured in one camera and the impact distance of the event measured
from the telescope (distance from the shower axis to the telescope). Up to now,
simple features have been used to feed regressors to estimate the
energy, e.g., the charge, the width and the length of the images, the impact
parameter and the height of the shower maximum.

Up to now, we used a Boosted Decision Tree (BDT) algorithm to reconstruct the
energy. Note that the tuning of the Random Forest (RF) algorithm was found
a bit problematic (20 % energy resolution at all energies). The important thing
to get the a good energy estimator is to build trees with high depth.
The minimal number of events to form an external node was fixed to 10 in order
to obtain a model with a reasonable size (~50MB for  for 200000 evts).
Indeed, allowing the trees to develop deeper would result in massive
files (~500MB for 200000 evts).

g/h classifier
--------------
To build a g/h classifier you need gamma-ray and proton tables with some
features discriminate between gamma and hadrons. Electrons are handled later
as a contamination (one could also train a classifier with gamma against
a background sample composed of weighted hadrons and weighted electrons).
The script `protopipe.scripts.build_model.py` takes as arguments a configuration
file:

.. code-block:: yaml

    General:
     model_type: 'classifier'  # regressor or classifier
     data_dir: 'absolute_data_path'  # Directory with the data
     data_sig_file: 'dl1_tail_gamma_merged.h5'  # Name of the signal file ({} will be completed with the mode (tail,wave))
     data_bkg_file: 'dl1_tail_proton_merged.h5'  # Name of the bkg file ({} will be completed with the mode (tail,wave))
     outdir: 'absolute_output_path'  # Output directory
     cam_id_list: ['LSTCam', 'NectarCam']  # List of camera
     table_name_template: 'feature_events_'  # Template name of table in DF5 (will be completed with cam_ids)

    Split:
     train_fraction: 0.8  # Fraction of events use to train the regressor
     use_same_number_of_sig_and_bkg_for_training: False  # Lowest statistics will drive the split

    Method:
     name: 'RandomForestClassifier'  # AdaBoostClassifier or RandomForestClassifier
     target_name: 'label'  # Name of the labels
     tuned_parameters:  # List of hyper-parameters to be optimized
      n_estimators: [200]
      max_depth: [null]
      min_samples_split: [2]
     scoring: 'roc_auc'  # Metrics to choose the best regressor
     cv: 2  # k in k-cross-validation
     use_proba: True  # If not, output is score
     calibrate_output: False  # If true calibrate probability (not tested)

    FeatureList:  # List of feature to build the g/h classifier
     - 'log10_reco_energy'
     - 'width'
     - 'length'
     - 'skewness'
     - 'kurtosis'
     - 'h_max'

    SigFiducialCuts:  # Fiducial cuts that will be applied on signal data
     - 'offset <= 0.5'

    BkgFiducialCuts:  # Fidicual cuts that will be applied on bkg data
     - 'offset <= 1.'

    Diagnostic:  #  For diagnostic plots
     energy:  # Energy binning (used for reco and true energy)
      nbins: 10
      min: 0.01
      max: 100

We want to exploit parameters showing statistical differences in the shower
developments between gamma-ray induced showers and hadron induced shower.
Up to now, we used the second moments of the images (width and length) as well
as the higer orders of the images (skewness and kurtosis which do not show a very high
separation power). We also use stereoscopic parameters such as the heigh of
the shower maximum and the reconstructed energy. The energy is important
since the distribution of the discriminant parameters vary a lot with
the energy of the particles.

Since in the end we want to average the score of the particles between different
cameras, we need the classifier to have an output normalised between 0 and 1.
Ideally, we would like also to get a `probabilistic classifier`_ (e.g. score of
0.8 gives a chance probability of 80 % that we are dealing with a signal event).
in order to average one pear with one pear (not an apple), but it's not so easy
since a lot a of cuts are done afterwards (angular cut, energy cut) which then
make the calibration caduc.

Anyway, we gave up on the BDT method since the output is not easy to normalise
between 0 and 1 (there are also fluctuations on the score distribution
that can totally crash the normalisation) and we trained a Random Forest (RF) as
people do the MARS analysis in CTA (not the same way as in MAGIC, e.g.
information of tel #1 and #2 in the same RF, here one model per type of telescope
then gammaness averaging).

Once again, the main important hyper-parameters
to get a robust classifier is the maximal depth of the trees and the
minimal number of events to get an external node (`min_samples_leaf`).
Please be aware that if you specify a `min_samples_leaf` close to one you'll be
in a high regime of overtraining that can be seen with an area under
the ROC (auc) of 1 for the training sample and a mismatch between the gammaness
distribution of the training and the test samples. In order to get an agreement
(by eye, could do a KS/chi2 test) between the training and test distributions
I chose to grow a forest of 200 trees with a max_depth of 10. I use a maximal
number of 200000 images for each sample for the training/test phase.

Note that the previous setup differ from what Abelardo is doing. Abelardo has
no max_depth, he grows 100 tress, and uses a min_samples_leaf close to 1 (TBC).
He is in an overtraining regime (auc ROC close to 1) and the agreement of the
distributions between the training and the test samples is bad. This is not good
since one might want to control the cut efficiencies of the models and
in real conditions to see that everything is correct.

*The settings used are not really optimised, I tuned them to get reasonable
performance and a good agreeement between the training/test samples.
Optimisation is welcome*

Diagnostics
-----------
To get diagnostic plots in order to control the robustness and the performance
of the models you can use the script `model_diagnostic.py`. It takes as arguments
a configuration file:

.. code-block:: bash

    >$ ./build_model.py --help
    usage: build_model.py [-h] --config_file CONFIG_FILE
                      [--wave | --tail]

For the energy estimator the diagnostic plots consist in:

* Distribution of the features
* Importance of the features
* Distribution of the ratio of the reconstructed energy over the true energy
  fitted with a gaussian for the subarrays
* Energy resolution and energy bias corresponding to the gaussian parametrisation
  for the subarrays

For a g/h classifier the following diagnostic are provided:

* Distribution of the features
* Importance of the features
* ROC curve (and its variation with energy)
* Output model distribution (and its variation with energy)

What could be improved?
=======================

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
=============

.. automodapi:: protopipe.mva
   :no-inheritance-diagram:
   :skip: auc, roc_curve

.. _scikit-learn: https://scikit-learn.org/
.. _GridSearchCV: https://scikit-learn.org/stable/modules/grid_search.html
.. _Pandas: https://pandas.pydata.org/
.. _probabilistic classifier: https://scikit-learn.org/stable/modules/calibration.html
