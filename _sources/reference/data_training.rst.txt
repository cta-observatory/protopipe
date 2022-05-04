.. _data_training:

Data training
=============

``protopipe.scripts.data_training`` is used to build tables of reconstructed image
and shower geometry parameters (that in *protopipe* constitutes generally the ``TRAINING`` format).
This type of data can be used to train energy and particle classification
estimators for each camera type.

.. note::
  | In the default analysis workflow, the particle classification model
    uses the estimate of the particle's energy as one of the parameters.
  | When training the data for that model you will need to specify the boolean
    ``estimate_energy`` parameter as well as the directory where the model is
    saved via the ``regressor_dir`` option.


By invoking the help argument, you can get help about how the script works:

.. code-block::

  usage: protopipe-TRAINING [-h] --config_file CONFIG_FILE -o OUTFILE [-m MAX_EVENTS] -i INDIR -f [INFILE_LIST [INFILE_LIST ...]]
                            [--cam_ids [CAM_IDS [CAM_IDS ...]]] [--wave_dir WAVE_DIR] [--wave_temp_dir WAVE_TEMP_DIR] [--wave | --tail]
                            [--debug] [--show_progress_bar] [--save_images] [--estimate_energy ESTIMATE_ENERGY]
                            [--regressor_dir REGRESSOR_DIR] [--regressor_config REGRESSOR_CONFIG]

  optional arguments:
    -h, --help            show this help message and exit
    --config_file CONFIG_FILE
    -o OUTFILE, --outfile OUTFILE
    -m MAX_EVENTS, --max_events MAX_EVENTS
                          maximum number of events considered per file
    -i INDIR, --indir INDIR
                          Input folder
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
    --show_progress_bar   Show information about execution progress
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
  # WARNING: the settings recorded here are unstable and used here only to give an example
  General:
    config_name: "ANALYSIS_NAME" # filled by the GRID interface
    production: "" # 'Prod3b' or 'Prod5N'
    site: "" # 'north' or 'south'
    # 'array' can be either
    # - a custom list of telescope IDs
    # - 'full_array', 'prod5N_alpha_north', 'prod5N_alpha_south', 'prod5N_alpha_south_NectarCam'
    # If a string, you can select a telescope-type subarray by
    # adding e.g. '_LST_LST_LSTCam'
    # NOTE: the 'full_array' is the total original array without any selection
    array: ""
    cam_id_list: [] # to upload the required models

  Calibration:
    apply_integration_correction: true # for CTAMARS-like analysis use false
    apply_peak_time_shift: false
    apply_waveform_time_shift: false
    # factor to transform the integrated charges (in ADC counts) into number of
    # photoelectrons (on top of the DC-to-PHE factor)
    # the pixel-wise one calculated by simtelarray is 0.92 for CTAMARS
    calib_scale: 1.0

  ImageCleaning: # NOTE: these are EXAMPLE values
    # Use only the biggest cluster of surviving pixels
    biggest:
      tail: # Cleaning based on the "tailcut" technique
        thresholds: # picture, boundary
          - LSTCam: [4.0, 2.0]
          - NectarCam: [4.0, 2.0]
          - FlashCam: [4, 2] # dummy values for reliable unit-testing
          - ASTRICam: [4, 2] # dummy values for reliable unit-testing
          - DigiCam: [0, 0] # values left unset for future studies
          - CHEC: [0, 0] # values left unset for future studies
          - SCTCam: [0, 0] # values left unset for future studies
        keep_isolated_pixels: False
        min_number_picture_neighbors: 1

      wave: # Cleaning based on the "wavelets" technique
        # Directory to write temporary files
        #tmp_files_directory: '/dev/shm/'
        tmp_files_directory: "./"
        options:
          LSTCam:
            type_of_filtering: "hard_filtering"
            filter_thresholds: [3, 0.2]
            last_scale_treatment: "drop"
            kill_isolated_pixels: True
            detect_only_positive_structures: False
            clusters_threshold: 0
          NectarCam: # TBC
            type_of_filtering: "hard_filtering"
            filter_thresholds: [3, 0.2]
            last_scale_treatment: "drop"
            kill_isolated_pixels: True
            detect_only_positive_structures: False
            clusters_threshold: 0

    # Use all clusters of surviving pixels
    extended:
      tail: # Cleaning based on the "tailcut" technique
        thresholds: # picture, boundary
          - LSTCam: [4.0, 2.0]
          - NectarCam: [4.0, 2.0]
          - FlashCam: [4, 2] # dummy values for reliable unit-testing
          - ASTRICam: [4, 2] # dummy values for reliable unit-testing
          - DigiCam: [0, 0] # values left unset for future studies
          - CHEC: [0, 0] # values left unset for future studies
          - SCTCam: [0, 0] # values left unset for future studies
        keep_isolated_pixels: False
        min_number_picture_neighbors: 1

      wave: # Cleaning based on the "wavelets" technique
        # Directory to write temporary files
        #tmp_files_directory: '/dev/shm/'
        tmp_files_directory: "./"
        options:
          LSTCam:
            type_of_filtering: "hard_filtering"
            filter_thresholds: [3, 0.2]
            last_scale_treatment: "posmask"
            kill_isolated_pixels: True
            detect_only_positive_structures: False
            clusters_threshold: 0
          NectarCam: # TBC
            type_of_filtering: "hard_filtering"
            filter_thresholds: [3, 0.2]
            last_scale_treatment: "posmask"
            kill_isolated_pixels: True
            detect_only_positive_structures: False
            clusters_threshold: 0

  # Image selection cuts
  # NOTE: these are EXAMPLE values
  ImageSelection:
    source: "extended" # biggest or extended
    charge: [50., 1e10]
    pixel: [3, 1e10]
    ellipticity: [0.1, 0.6]
    nominal_distance: [0., 0.8] # in camera radius

  # Minimal number of telescopes to consider events
  Reconstruction:
    # for events with <2 LST images the single-LST image is removed
    # before shower geometry
    LST_stereo: True
    # after this we check if the remaining images satisfy the min_tel condition
    min_tel: 2 # any tel_type

  # Parameters for energy estimation
  EnergyRegressor:
    # Name of the regression method (e.g. AdaBoostRegressor, etc.)
    method_name: "RandomForestRegressor"
    estimation_weight: "CTAMARS" # CTAMARS == 1/RMS^2 (RMS from the RF trees)

  # Parameters for g/h separation
  GammaHadronClassifier:
    # Name of the classification method (e.g. AdaBoostRegressor, etc.)
    method_name: "RandomForestClassifier"
    # Use probability output or score
    use_proba: True
    estimation_weight: "hillas_intensity**0.54" # empirical value from CTAMARS
