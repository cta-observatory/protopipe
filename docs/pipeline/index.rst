.. _pipeline:

=====================
Pipeline (`pipeline`)
=====================

Introduction
============
`protopipe.pipeline` contains classes that are used in scripts
producing tables with images information (DL1), typically for g/h classifier and
energy regressor, and tables with event information, typically used for
performance estimation (DL2).


How to process events
=====================
Two scripts are used in `protopipe.scripts` to process events up to a DL1 and a DL2 level:
`protopipe.scripts.write_dl1.py` and `protopipe.scripts.write_dl2.py`, respectively.
Both scripts use a YAML configuration file and a set of optional command line
arguments.

An example of configuration file is shown below:

.. configuration-block::

    .. code-block:: yaml
        # General informations
        General:
         config_name: 'prod_full_array_north_zen20_az0'
         site: 'north'  # North or South
         array: 'full_array'  # subarray_LSTs, subarray_MSTs, full_array
         cam_id_list: ['LSTCam', 'NectarCam'] # Camera identifiers

        # Cleaning for reconstruction
        ImageCleaning:

         # Cleaning for reconstruction
         biggest:
          tail:  #
           thresholds:  # picture, boundary
            - LSTCam: [6, 3]
            - NectarCam: [6, 3]  # TBC
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
          tail:  #
           thresholds:  # picture, boundary
            - LSTCam: [6, 3]
            - NectarCam: [6, 3]  # TBC
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
         min_tel: 3



Reference/API
=============

.. automodapi:: protopipe.pipeline
   :no-inheritance-diagram:
