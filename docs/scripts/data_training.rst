.. _data_training:

Data training
=============

``protopipe.scripts.data_training`` is used to build tables of image and shower
parameters that will be further used to train energy and particle classification
estimators for each camera type.

.. note::
  | In the current version of the pipeline, the particle classification model
    needs uses the estimate of the particle's energy as one of the parameters.
  | When training the data for that model you will need to specify the boolean
    ``estimate_energy`` parameter as well as the directory where the model is
    saved via the ``regressor_dir`` option.


By invoking the help argument, you can get help about how the script works:

.. code-block::

  usage: protopipe-TRAINING [-h] --config_file CONFIG_FILE -o OUTFILE [-m MAX_EVENTS] [-i INDIR] [-f [INFILE_LIST [INFILE_LIST ...]]]
                          [--cam_ids [CAM_IDS [CAM_IDS ...]]] [--wave_dir WAVE_DIR] [--wave_temp_dir WAVE_TEMP_DIR] [--wave | --tail]
                          [--debug] [--save_images] [--estimate_energy ESTIMATE_ENERGY] [--regressor_dir REGRESSOR_DIR]
                          [--regressor_config REGRESSOR_CONFIG]

  optional arguments:
    -h, --help            show this help message and exit
    --config_file CONFIG_FILE
    -o OUTFILE, --outfile OUTFILE
    -m MAX_EVENTS, --max_events MAX_EVENTS
                          maximum number of events considered per file
    -i INDIR, --indir INDIR
    -f [INFILE_LIST [INFILE_LIST ...]], --infile_list [INFILE_LIST [INFILE_LIST ...]]
                          give a specific list of files to run on
    --cam_ids [CAM_IDS [CAM_IDS ...]]
                          give the specific list of camera types to run on
    --wave_dir WAVE_DIR   directory where to find mr_filter. if not set look in $PATH
    --wave_temp_dir WAVE_TEMP_DIR
                          directory where mr_filter to store the temporary fits files
    --wave                if set, use wavelet cleaning -- default
    --tail                if set, use tail cleaning, otherwise wavelets
    --debug               Print debugging information
    --save_images         Save also all images
    --estimate_energy ESTIMATE_ENERGY
                          Estimate the events' energy with a regressor from protopipe.scripts.build_model
    --regressor_dir REGRESSOR_DIR
                          regressors directory
    --regressor_config REGRESSOR_CONFIG
                          Configuration file used to produce regressor model

The configuration file used by this script is ``analysis.yaml``,

.. code-block:: yaml

  # General informations
  # NOTE: only Prod3b simulations are currently supported.
  General:
  config_name: 'v0.4.0_dev1'
  site: 'north'  # 'north' or 'south'
  # array can be either
  # - 'subarray_LSTs', 'subarray_MSTs', 'subarray_SSTs' or 'full_array'
  # - a custom list of telescope IDs
  # WARNING: for simulations containing multiple copies of the telescopes,
  # only 'full_array' or custom list are supported options!
  array: full_array
  cam_id_list : ['LSTCam', 'NectarCam'] # List of camera IDs to be used

  # Cleaning for reconstruction
  ImageCleaning:

  # Cleaning for reconstruction
  biggest:
  tail:  #
   thresholds:  # picture, boundary
    - LSTCam: [6.61, 3.30]  # TBC
    - NectarCam: [5.75, 2.88]  # TBC
    - FlashCam: [4,2] # dummy values for reliable unit-testing
    - ASTRICam: [4,2] # dummy values for reliable unit-testing
    - DigiCam: [0,0] # values left unset for future studies
    - CHEC: [0,0] # values left unset for future studies
    - SCTCam: [0,0] # values left unset for future studies
   keep_isolated_pixels: False
   min_number_picture_neighbors: 1

  wave:
   # Directory to write temporary files
   #tmp_files_directory: '/dev/shm/'
   tmp_files_directory: './'
   options:
    LSTCam:
     type_of_filtering: 'hard_filtering'
     filter_thresholds: [3, 0.2]
     last_scale_treatment: 'drop'
     kill_isolated_pixels: True
     detect_only_positive_structures: False
     clusters_threshold: 0
    NectarCam:  # TBC
     type_of_filtering: 'hard_filtering'
     filter_thresholds: [3, 0.2]
     last_scale_treatment: 'drop'
     kill_isolated_pixels: True
     detect_only_positive_structures: False
     clusters_threshold: 0

  # Cleaning for energy/score estimation
  extended:
  tail:  #
   thresholds:  # picture, boundary
    - LSTCam: [6.61, 3.30]  # TBC
    - NectarCam: [5.75, 2.88]  # TBC
    - FlashCam: [4,2] # dummy values for reliable unit-testing
    - ASTRICam: [4,2] # dummy values for reliable unit-testing
    - DigiCam: [0,0] # values left unset for future studies
    - CHEC: [0,0] # values left unset for future studies
    - SCTCam: [0,0] # values left unset for future studies
   keep_isolated_pixels: False
   min_number_picture_neighbors: 1

  wave:
   # Directory to write temporary files
   #tmp_files_directory: '/dev/shm/'
   tmp_files_directory: './'
   options:
    LSTCam:
     type_of_filtering: 'hard_filtering'
     filter_thresholds: [3, 0.2]
     last_scale_treatment: 'posmask'
     kill_isolated_pixels: True
     detect_only_positive_structures: False
     clusters_threshold: 0
    NectarCam:  # TBC
     type_of_filtering: 'hard_filtering'
     filter_thresholds: [3, 0.2]
     last_scale_treatment: 'posmask'
     kill_isolated_pixels: True
     detect_only_positive_structures: False
     clusters_threshold: 0

  # Cut for image selection
  ImageSelection:
  charge: [50., 1e10]
  pixel: [3, 1e10]
  ellipticity: [0.1, 0.6]
  nominal_distance: [0., 0.8]  # in camera radius

  # Minimal number of telescopes to consider events
  Reconstruction:
  min_tel: 2

  # Parameters for energy estimation
  EnergyRegressor:
  # Name of the regression method (e.g. AdaBoostRegressor, etc.)
  method_name: 'AdaBoostRegressor'

  # Parameters for g/h separation
  GammaHadronClassifier:
  # Name of the classification method (e.g. AdaBoostRegressor, etc.)
  method_name: 'RandomForestClassifier'
  # Use probability output or score
  use_proba: True