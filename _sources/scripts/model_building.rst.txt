.. _model_building:

Building the models
===================

With *protopipe* it is possible to build estimation models for both particle
energy and gamma/hadron classification.
The base classes are defined in the ``protopipe.mva`` module (see :ref:`mva`).

For both cases, building a regressor or a classifier, the script
``protopipe.scripts.build_model.py`` is used.

The following is a summary of ist usage required arguments and options.

.. code-block::

    >$ ./build_model.py --help
    usage: build_model.py [-h] --config_file CONFIG_FILE [--max_events MAX_EVENTS]
                      [--wave | --tail]

    Build model for regression/classification

    optional arguments:
      -h, --help            show this help message and exit
      --config_file         CONFIG_FILE
      --max_events          maximum number of events for training
      --wave                if set, use wavelet cleaning
      --tail                if set, use tail cleaning, otherwise wavelets

The script takes along its arguments a configuration file which depends on what
type of estimator needs to be trained:

* ``regressor.yaml`` is used to train an energy regressor,
* ``classifier.yaml`` is used to train a gamma/hadron classifier.

Energy regressor
----------------

To build this you need a table with at least the MC energy (the target)
and some event characteristics (the features) to reconstruct the energy.
This table is created in the :ref:`data_training` step.

The following is a commented example of the required configuration file
``regressor.yaml``:

.. code-block:: yaml

    General:
     model_type: 'regressor'  # regressor or classifier
     data_dir: 'absolute_data_path'  # Directory with the data
     data_file: 'dl1_{}_gamma_merged.h5'  # Name of the data file ({} will be completed with the mode (tail,wave))
     outdir: 'absolute_output_path'  # Output directory
     cam_id_list: ['LSTCam', 'NectarCam']  # List of the cameras to be used
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

To estimate the energy of a gamma-ray, one wants to use the relation between
the charge measured in one camera and the impact distance of the event measured
from the telescope (distance from the shower axis to the telescope).

Up to now, simple features have been used to feed regressors to estimate the
energy, e.g.,

* the charge,
* the width and the length of the images,
* the impact parameter,
* the height of the shower maximum.

The algorithm used to reconstruct the energy is a **Boosted Decision Tree (BDT)**.
The tuning of the *Random Forest (RF)* algorithm was found to be
a bit problematic (20 % energy resolution at all energies).
The important thing to get a good energy estimator is to build trees with high
depth.
The minimal number of events to form an external node was fixed to 10 in order
to obtain a model with a reasonable size (~50MB for  for 200000 evts).
Indeed, allowing the trees to develop deeper would result in massive
files (~500MB for 200000 evts).

g/h classifier
--------------

To build a gamma/hadron classifier you need gamma-ray and proton tables with some
features used to discriminate between gamma and hadrons (electrons are handled later
as a contamination).

.. note::
  An alternative approach - yet to study - could be to train a classifier with gamma
  against a background sample composed of weighted hadrons and weighted electrons.

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
as the higher orders of the images (skewness and kurtosis which do not show a very high
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

.. warning::

  The default settings used are not yet optimised for every case.

  They have been tuned to get reasonable performance and a good agreeement
  between the training/test samples.

  A first optimisation will follow from the comparison against CTA-MARS, even
  though the parameters used and settings are already the same.

.. _probabilistic classifier: https://scikit-learn.org/stable/modules/calibration.html
