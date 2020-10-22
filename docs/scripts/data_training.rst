.. _data_training:

Data training
=============

`protopipe.scripts.write_dl1.py` is used to build tables of image and stereo
parameters that will be further used to build energy estimators and
classifiers per type of camera.
In case you want to estimate the energy of the particles,
to build g/h classifiers, typically, you need to specify the boolean
`estimate_energy` parameter as well as the directory where the model is
saved via the `regressor_dir` option.

By invoking the help argument, you can get help about how the script works:

.. code-block::

    usage: write_dl1.py [-h] --config_file CONFIG_FILE -o OUTFILE [-m MAX_EVENTS]
                        [-i INDIR] [-f [INFILE_LIST [INFILE_LIST ...]]]
                        [--wave_dir WAVE_DIR] [--wave_temp_dir WAVE_TEMP_DIR]
                        [--wave | --tail] [--save_images] [--estimate_energy]
                        [--regressor_dir REGRESSOR_DIR]

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
      --save_images         Save images in images.h5 (one file testing)
      --estimate_energy     Estimate energy
      --regressor_dir REGRESSOR_DIR
                            regressors directory
