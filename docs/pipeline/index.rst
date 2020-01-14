.. _pipeline:

========
pipeline
========

Introduction
============
`protopipe.pipeline` contains classes that are used in scripts
producing tables with images information (DL1), typically for g/h classifier and
energy regressor, and tables with event information, typically used for
performance estimation (DL2).

Two classes from the sub-module are used to process the events. The EventPreparer class, which
goal is to loop on events and to provide event parameters (e.g. impact parameter) and
image parameters (e.g. Hillas parameters). The second class, ImageCleaner,
is dedicated to clean the images according to different options
(tail cut and/or wavelet).

The tables are written on disk in HDF5_ format using the PyTables_
Python module. This choice has been done considering the potential huge
volume of data we will be dealing with in a near future.

How to process events
=====================
Two scripts are used in `protopipe.scripts` to process events up to a DL1 and a DL2 level:
`protopipe.scripts.write_dl1.py` and `protopipe.scripts.write_dl2.py`, respectively.
Both scripts use a YAML configuration file and a set of optional command line
arguments. You can get help about how the scripts work by invoking the help argument:

.. code-block:: bash

    >$ ./write_dl1.py --help
    usage: write_dl1.py [-h] --config_file CONFIG_FILE -o OUTFILE [-m MAX_EVENTS]
                        [-i INDIR] [-f [INFILE_LIST [INFILE_LIST ...]]]
                        [--cam_ids [CAM_IDS [CAM_IDS ...]]] [--wave_dir WAVE_DIR]
                        [--wave_temp_dir WAVE_TEMP_DIR] [--wave | --tail]
                        [--estimate_energy ESTIMATE_ENERGY]
                        [--regressor_dir REGRESSOR_DIR]

In both scripts you need to specify an input file (or a list) as well as
its associated directory, a list of camera IDs in order to inform the
programs of which tables you want to build (and also to load the needed
models). In addition you need to specify the type of cleaning to processing
the images (`tail` or `wave`). For the wavelet method you also need to define
the repository to save temporary results with the `wave_temp_dir` arguments.

How to produce DL1+ data
------------------------

`protopipe.scripts.write_dl1.py` is used to build tables of image and stereo
parameters that will be further used to build energy estimators and
classifiers per type of camera.
In case you want to estimate the energy of the particles,
to build g/h classifiers, typically, you need to specify the boolean
`estimate_energy` parameter as well as the directory where the model is
saved via the `regressor_dir` option.

How to produce DL2 data
-----------------------

`protopipe.scripts.write_dl2.py` is used to produce DL2 tables with the event
caracteristics such as the direction, the energy and the score/gammaness.
You will need to specify the locations of the models for the energy and
gammaness estimations.

Configuration file
==================
Both scripts are used with a YAML configuration file.
A configuration file example  with some comments is shown below:

.. code-block:: yaml

    # General informations
    General:
     config_name: 'prod_full_array_north_zen20_az0_complete'
     site: 'north'  # North or South
     array: 'full_array'  # subarray_LSTs, subarray_MSTs, full_array
     cam_id_list: ['LSTCam', 'NectarCam'] # Camera identifiers (Should be read in scripts?)

    # Cleaning for reconstruction
    ImageCleaning:

     # Cleaning for reconstruction
     biggest:
      tail:
       thresholds:  # picture, boundary
        - LSTCam: [6, 3]
        - NectarCam: [6, 3]
       keep_isolated_pixels: False
       min_number_picture_neighbors: 2

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
      tail:
       thresholds:  # picture, boundary
        - LSTCam: [6, 3]
        - NectarCam: [6, 3]
       keep_isolated_pixels: False
       min_number_picture_neighbors: 2

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
     # Name of the regression method (e.g. AdaBoostRegressor, None, etc.)
     method_name: 'AdaBoostRegressor'

    # Parameters for g/h separation
    GammaHadronClassifier:
     # Name of the classification method (e.g. AdaBoostRegressor, None, etc.)
     method_name: 'RandomForestClassifier'
     # Use probability output or score
     use_proba: True

Note that you can process events without estimations of the energy or the
score of the particles if you specify the `method_name` as `None` in the
corresponding sections.

What could be improved?
=======================

* Saving calibrated images as a DL1 format (here, only image parameters are saved).
  It should simplify analysis production, e.g. having DL1 calibrated data on disk allows to study
  the impact on direction reconstruction of several cleaning
  methods by processing the simtel files only once. Huge gain of time!
* The EventPreparer is a bit messy, it should returns the event and one container
  with several results (hillas parameters, reconstructed shower, etc.). In addition
  some things are hard-coded , e.g. for now calibration is done in the same way
  (not a problem since only LSTCam and NectarCam have been considered until now),
  camera radius is also hard-coded for LST and MST, and computation of the impact
  parameters in the frame of the shower system should be better implemented.
* Features for regression and classification are hard-coded in the two scripts
  `write_dl1.py` and `write_dl2.py`
* The fields of the output tables do not follow any DL1 or DL2
  data format specified in the `gamma astro data format`_ initiative

To be fixed
===========

* Why specify the cam_id in the scripts writting the tables (DL1 or DL2)
  although it is already given in the configuration file?

Reference/API
=============

.. automodapi:: protopipe.pipeline
   :no-inheritance-diagram:
   :skip: event_source

.. _PyTables: https://www.pytables.org/
.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
.. _gamma astro data format: https://gamma-astro-data-formats.readthedocs.io/en/latest/index.html
