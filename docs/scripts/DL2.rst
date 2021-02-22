.. _DL2:

Production of DL2 data
======================

`protopipe.scripts.write_dl2.py` is used to produce DL2 tables labeled with
shower information such as the direction, the energy and the score/gammaness.
You will need to specify the locations of the models for the energy and
gammaness estimations created in the :ref:`model_building` step.

The configuration file used by this script ``analysis.yaml``, the same as for
``protopipe.scripts.data_training``.

By invoking the help argument, you can get help about how the script works:

.. code-block::

  usage: write_dl2.py [-h] --config_file CONFIG_FILE -o OUTFILE [-m MAX_EVENTS]
                    [-i INDIR] [-f [INFILE_LIST [INFILE_LIST ...]]]
                    [--wave_dir WAVE_DIR] [--wave_temp_dir WAVE_TEMP_DIR]
                    [--wave | --tail] [--regressor_dir REGRESSOR_DIR]
                    [--classifier_dir CLASSIFIER_DIR]
                    [--force_tailcut_for_extended_cleaning FORCE_TAILCUT_FOR_EXTENDED_CLEANING]
                    [--save_images]

  optional arguments:
    -h, --help            show this help message and exit
    --config_file CONFIG_FILE
    -o OUTFILE, --outfile OUTFILE
    -m MAX_EVENTS, --max_events MAX_EVENTS
                          maximum number of events considered per file
    -i INDIR, --indir INDIR
    -f [INFILE_LIST [INFILE_LIST ...]], --infile_list [INFILE_LIST [INFILE_LIST ...]]
                          give a specific list of files to run on
    --wave_dir WAVE_DIR   directory where to find mr_filter. if not set look in
                          $PATH
    --wave_temp_dir WAVE_TEMP_DIR
                          directory where mr_filter to store the temporary fits
                          files
    --wave                if set, use wavelet cleaning -- default
    --tail                if set, use tail cleaning, otherwise wavelets
    --regressor_dir REGRESSOR_DIR
                          regressors directory
    --classifier_dir CLASSIFIER_DIR
                          regressors directory
    --force_tailcut_for_extended_cleaning FORCE_TAILCUT_FOR_EXTENDED_CLEANING
                          For tailcut cleaning for energy/score estimation
    --save_images         Save images in images.h5 (one file testing)
