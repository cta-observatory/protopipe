.. _model_building:

Building the models
===================

With *protopipe* it is possible to build estimation models for both particle
energy and gamma/hadron classification.
The base classes are defined in the ``protopipe.mva`` module (see :ref:`mva`).

For both cases, building a regressor or a classifier, the script
``protopipe.scripts.build_model.py`` is used.

The following is the help output which shows required arguments and options.

.. code-block::

    >$ protopipe-MODEL -h
    usage: protopipe-MODEL [-h] --config_file CONFIG_FILE [--max_events MAX_EVENTS] [--wave | --tail]
                       (--cameras_from_config | --cameras_from_file | --cam_id_list CAM_ID_LIST) [-i INDIR] [--infile_signal INFILE_SIGNAL]
                       [--infile_background INFILE_BACKGROUND] [-o OUTDIR]

    Build model for regression/classification

    optional arguments:
      -h, --help            show this help message and exit
      --config_file CONFIG_FILE
      --max_events MAX_EVENTS
                            maximum number of events to use
      --wave                if set, use wavelet cleaning
      --tail                if set, use tail cleaning (default), otherwise wavelets
      --cameras_from_config
                            Get cameras configuration file (Priority 1)
      --cameras_from_file   Get cameras from input file (Priority 2)
      --cam_id_list CAM_ID_LIST
                            Select cameras like 'LSTCam CHEC' (Priority 3)
      -i INDIR, --indir INDIR
                            Directory containing the required input file(s)
      --infile_signal INFILE_SIGNAL
                            SIGNAL file (default: read from config file)
      --infile_background INFILE_BACKGROUND
                            BACKGROUND file (default: read from config file)
      -o OUTDIR, --outdir OUTDIR

The script takes along its arguments a configuration file which depends on what
type of model needs to be built.

The available choices can be found under ```protopipe.aux.example_config_files```:

* ``AdaBoostRegressor.yaml`` is used to train an energy regressor,
* ``RandomForestRegressor.yaml``  is used to train an energy regressor,
* ``RandomForestClassifier.yaml`` is used to train a gamma/hadron classifier.

Energy regressor
----------------

To build this you need a table with at least the MC energy (the target)
and some event characteristics (the features) to reconstruct the energy.
This table is created in the :ref:`data_training` step.

The following is a commented example of the required configuration file
``AdaBoostRegressor.yaml`` with similar options as for ``RandomForestRegressor.yaml``,

.. code-block:: yaml

  General:
    # [...] = your analysis local full path OUTSIDE the Vagrant box
    data_dir: '../../data/'
    data_sig_file: 'TRAINING_energy_tail_gamma_merged.h5'
    outdir: './'
    
    # List of cameras to use (you can override this from the CLI)
    cam_id_list: ['LSTCam', 'NectarCam']

  # If train_fraction is 1, all the TRAINING dataset will be used to train the
  # model and benchmarking can only be done from the benchmarking notebook
  # TRAINING/benchmarks_DL2_to_classification.ipynb
  Split:
    train_fraction: 0.8
    use_same_number_of_sig_and_bkg_for_training: False  # Lowest statistics will drive the split

  # Optimize the hyper-parameters of the estimator with a grid search
  # If True parameters should be provided as lists
  # If False the model used will be the unique one based on your the
  GridSearchCV:
    use: False # True or False
    # if False the following two variables are irrelevant
    scoring: 'explained_variance'
    cv: 2

  Method:
    name: 'sklearn.ensemble.AdaBoostRegressor'
    target_name: 'true_energy'
    # Please, see scikit-learn's API for what each parameter means
    # NOTE: null == None
    base_estimator:
      name: 'sklearn.tree.DecisionTreeRegressor'
      parameters:
        # NOTE: here we set the parameters relevant for sklearn.tree.DecisionTreeRegressor
        criterion: "mse" # "mse", "friedman_mse", "mae" or "poisson"
        splitter: "best" # "best" or "random"
        max_depth: null # null or integer
        min_samples_split: 2 # integer or float
        min_samples_leaf: 1 # int or float
        min_weight_fraction_leaf: 0.0 # float
        max_features: null # null, "auto", "sqrt", "log2", int or float
        max_leaf_nodes: null # null or integer
        min_impurity_decrease: 0.0 # float
        random_state: 0 # null or integer or RandomState
        ccp_alpha: 0.0 # non-negative float
    tuned_parameters:
      n_estimators: 50
      learning_rate: 1
      loss: 'linear' # 'linear', 'square' or 'exponential'
      random_state: 0 # int, RandomState instance or None

  # List of the features to use to train the model
  # You can:
  # - comment/uncomment the ones you see here,
  # - add new ones here if they can be evaluated with pandas.DataFrame.eval
  # - if not you can propose modifications to protopipe.mva.utils.prepare_data
  FeatureList:
    Basic: # single-named, they need to correspond to input data columns
    - 'h_max'         # Height of shower maximum from stereoscopic reconstruction
    - 'impact_dist'   # Impact parameter from stereoscopic reconstruction
    - 'hillas_width'  # Image Width
    - 'hillas_length' # Image Length
    # - 'concentration_pixel' # Percentage of photo-electrons in the brightest pixel
    - 'leakage_intensity_width_1_reco' # fraction of total Intensity which is contained in the outermost pixels of the camera
    Derived: # custom evaluations of basic features that will be added to the data
      # column name : expression to evaluate using basic column names
      log10_WLS: log10(hillas_width*hillas_length/hillas_intensity)
      log10_intensity: log10(hillas_intensity)
      CTAMARS_1: (sqrt((hillas_x - az)**2 + (hillas_y - alt)**2))**2
      CTAMARS_2: arctan2(hillas_y - alt, hillas_x - az)

  # These cuts select the input data BEFORE training
  SigFiducialCuts:
    - 'good_image == 1'
    - 'is_valid == True'
    - 'hillas_intensity_reco > 0'

  Diagnostic:
   # Energy binning (used for reco and true energy)
   energy:
    nbins: 15
    min: 0.0125
    max: 125

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

The following the example provided by the example configuration file ``RandomForestClassifier.yaml``,

.. code-block:: yaml

  General:
    # [...] = your analysis local full path OUTSIDE the Vagrant box
    data_dir: '../../data/' # '[...]/data/TRAINING/for_particle_classification/'
    data_sig_file: 'TRAINING_classification_tail_gamma_merged.h5'
    data_bkg_file: 'TRAINING_classification_tail_proton_merged.h5'
    outdir: './' # [...]/estimators/gamma_hadron_classifier
    
    # List of cameras to use (protopipe-MODEL help output for other options)
    cam_id_list: ['LSTCam', 'NectarCam']

  # If train_fraction is 1, all the TRAINING dataset will be used to train the
  # model and benchmarking can only be done from the benchmarking notebook
  # TRAINING/benchmarks_DL2_to_classification.ipynb
  Split:
    train_fraction: 0.8
    use_same_number_of_sig_and_bkg_for_training: False  # Lowest statistics will drive the split

  # Optimize the hyper-parameters of the estimator with a grid search
  # If 'True' parameters should be provided as lists (for None use [null])
  # If 'False' the model used will be the unique one based on your the
  GridSearchCV:
    use: False # 'True' or 'False'
    # if False the following to variables are irrelevant
    scoring: 'roc_auc'
    cv: 2

  # Definition of the algorithm/method used and its hyper-parameters
  Method:
    name: 'sklearn.ensemble.RandomForestClassifier' # DO NOT CHANGE
    target_name: 'label' # defined between 0 and 1 (DO NOT CHANGE)
    tuned_parameters:
      # Please, see scikit-learn's API for what each parameter means
      # WARNING: null (not a string) == 'None'
      n_estimators: 100 # integer
      criterion: 'gini' # 'gini' or 'entropy'
      max_depth: null # null or integer
      min_samples_split: 2 # integer or float
      min_samples_leaf: 1 # integer or float
      min_weight_fraction_leaf: 0.0 # float
      max_features: 3 # 'auto', 'sqrt', 'log2', integer or float
      max_leaf_nodes: null # null or integer
      min_impurity_decrease: 0.0 # float
      bootstrap: False # True or False
      oob_score: False # True or False
      n_jobs: null # null or integer
      random_state: 0 # null or integer or RandomState
      verbose: 0 # integer
      warm_start: False # 'True' or 'False'
      class_weight: null # 'balanced', 'balanced_subsample', null, dict or list of dicts
      ccp_alpha: 0.0 # non-negative float
      max_samples: null # null, integer or float
    calibrate_output: False  # If True calibrate model on test data

  # List of the features to use to train the model
  # You can:
  # - comment/uncomment the ones you see here,
  # - add new ones here if they can be evaluated with pandas.DataFrame.eval
  # - if not you can propose modifications to protopipe.mva.utils.prepare_data
  FeatureList:
    Basic: # single-named, they need to correspond to input data columns
    - 'h_max'         # Height of shower maximum from stereoscopic reconstruction
    - 'impact_dist'   # Impact parameter from stereoscopic reconstruction
    - 'hillas_width'  # Image Width
    - 'hillas_length' # Image Length
    # - 'concentration_pixel' # Percentage of photo-electrons in the brightest pixel
    Derived: # custom evaluations of basic features that will be added to the data
      # column name : expression to evaluate using basic column names
      log10_intensity: log10(hillas_intensity)
      log10_energy: log10(reco_energy) # Averaged-estimated energy of the shower
      log10_energy_tel: log10(reco_energy_tel) # Estimated energy of the shower per telescope

  # These cuts select the input data BEFORE training
  SigFiducialCuts:
    - 'good_image == 1'
    - 'is_valid == True'
    - 'hillas_intensity_reco > 0'

  BkgFiducialCuts:
   - 'good_image == 1'
   - 'is_valid == True'
   - 'hillas_intensity_reco > 0'

  Diagnostic:
   # Energy binning (used for reco and true energy)
   energy:
    nbins: 4
    min: 0.0125
    max: 200

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
