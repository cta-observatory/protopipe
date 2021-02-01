.. _data_training:

Data training
=============

``protopipe.scripts.data_training`` is used to build tables of image and shower
parameters that will be further used to build energy and particle classification
estimators for camera type.

.. note::
  | In the current version of the pipeline, the particle classification model
    needs uses the estimate of the particle's energy as one of the parameters.
  | When training the data for that model you will need to specify the boolean
    ``estimate_energy`` parameter as well as the directory where the model is
    saved via the ``regressor_dir`` option.


By invoking the help argument, you can get help about how the script works:

.. code-block::

  usage: data_training.py [-h] --config_file CONFIG_FILE -o OUTFILE
                      [-m MAX_EVENTS] [-i INDIR]
                      [-f [INFILE_LIST [INFILE_LIST ...]]]
                      [--cam_ids [CAM_IDS [CAM_IDS ...]]]
                      [--wave_dir WAVE_DIR] [--wave_temp_dir WAVE_TEMP_DIR]
                      [--wave | --tail] [--debug] [--save_images]
                      [--estimate_energy ESTIMATE_ENERGY]
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
  --cam_ids [CAM_IDS [CAM_IDS ...]]
                        give the specific list of camera types to run on
  --wave_dir WAVE_DIR   directory where to find mr_filter. if not set look in
                        $PATH
  --wave_temp_dir WAVE_TEMP_DIR
                        directory where mr_filter to store the temporary fits
                        files
  --wave                if set, use wavelet cleaning -- default
  --tail                if set, use tail cleaning, otherwise wavelets
  --debug               Print debugging information
  --save_images         Save also all images
  --estimate_energy ESTIMATE_ENERGY
                        Estimate the events' energy with a regressor from
                        protopipe.scripts.build_model
  --regressor_dir REGRESSOR_DIR
                        regressors directory
