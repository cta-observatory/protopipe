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
      loss: 'linear' # 'linear', 'square' or 'exponential'
      random_state: 0 # int, RandomState instance or None

  # List of the features to use to train the model
  # You can:
  # - comment/uncomment the ones you see here,
  # - add new ones here if they can be evaluated with pandas.DataFrame.eval
  # - if not you can propose modifications to protopipe.mva.utils.prepare_data
  FeatureList:
    Basic: # single-named, they need to correspond to input data columns
    - 'h_max'         # Height of shower maximum from stereoscopic reconstruction
    - 'impact_dist'   # Impact parameter from stereoscopic reconstruction
    - 'hillas_width'  # Image Width
    - 'hillas_length' # Image Length
    # - 'concentration_pixel' # Percentage of photo-electrons in the brightest pixel
    - 'leakage_intensity_width_1_reco' # fraction of total Intensity which is contained in the outermost pixels of the camera
    Derived: # custom evaluations of basic features that will be added to the data
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
    Basic: # single-named, they need to correspond to input data columns
    - 'h_max'         # Height of shower maximum from stereoscopic reconstruction
    - 'impact_dist'   # Impact parameter from stereoscopic reconstruction
    - 'hillas_width'  # Image Width
    - 'hillas_length' # Image Length
    # - 'concentration_pixel' # Percentage of photo-electrons in the brightest pixel
    Derived: # custom evaluations of basic features that will be added to the data
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

.. warning::

  The default settings used are not yet optimised for every case.

  They have been tuned to get reasonable performance and a good agreeement
  between the training/test samples.

  A first optimisation will follow from the comparison against CTA-MARS, even
  though the parameters used and settings are already the same.