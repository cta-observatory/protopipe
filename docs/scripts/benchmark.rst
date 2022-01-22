.. _benchmark:

Benchmarking
============

``protopipe-BENCHMARK`` is used to run benchmarks within *protopipe*.
It allows to look for available benchmarking notebooks, interface with them
and optionally convert them in HTML format for easier consultation.

.. warning::
  
  This is not currently available for the *calibration* benchmark notebook
  which requires a *ctapipe* version more recent than the one which protopipe
  supports.
  It will behave as the rest as soon as we upgrade to the latest release.

By invoking the help argument, you can get help about how the script works:
``protopipe-BENCHMARK`` allows for 2 commands ``list`` and ``launch``,

.. code-block::

  usage: protopipe-BENCHMARK [-h] {list,launch} ...

        Launch a benchmark notebook and convert it to an HTML page.
        USAGE EXAMPLE:
        --------------
        >>> protopipe-BENCHMARK list
        >>> protopipe-BENCHMARK launch -n TRAINING/benchmarks_DL1b_image-cleaning --config_file benchmarks.yaml


  positional arguments:
    {list,launch}
      list         List available benchmarks
      launch       Launch a specific benchmark

  optional arguments:
    -h, --help     show this help message and exit

In particular, the ``launch`` command is essentially a convenient wrapper
around `papermill <https://papermill.readthedocs.io/en/latest/>`__ and
(optionally) `jupyter nbconvert <https://nbconvert.readthedocs.io/en/latest/>`__.

.. code-block::

  usage: protopipe-BENCHMARK launch [-h] [--help-notebook] -n NAME --config_file CONFIG_FILE [-k [KWARGS [KWARGS ...]]] [--outpath OUTPATH]
                                    [--overwrite] [--suffix SUFFIX] [--no_export]

  optional arguments:
    -h, --help            show this help message and exit
    --help-notebook       Print the list of available notebook parameters
    -n NAME, --name NAME  Pipeline step and name of the benchmark (for a list use `protopipe-BENCHMARK -l`)
    --config_file CONFIG_FILE
                          Configuration file (default: stored under analysis 'config' folder)
    -k [KWARGS [KWARGS ...]], --kwargs [KWARGS [KWARGS ...]]
                          Overwrite or specify other configuration options (e.g. --kwargs foo=bar fiz=biz)
    --outpath OUTPATH     If unset it will be read from benchmaks.yaml
    --overwrite           Execute the notebook even if it overwrites the old result.
    --suffix SUFFIX       Suffix for result and HTML files (default: analysis name)
    --no_export           Do not convert the result notebook to any other format.

The configuration file used by this script is ``benchmarks.yaml``,

.. code-block:: yaml

  # This configuration file simplifies the usage of benchmarks throughout the
  # entire analysis.
  # It is recommended to fill it and specify any remaining options using
  # the --kwargs flag of protopipe-BENCHMARKS
  # To specify directories, please provide full paths
  # Note: users which use a CTADIRAC container should use paths OUTSIDE of it

  # General settings for you analysis
  analyses_directory: "ANALYSES_DIRECTORY" # filled by the grid interface
  analysis_name: "ANALYSIS_NAME" # filled by the grid interface
  # to compare with a previous release or version
  load_protopipe_previous: False # If True load data from a previous analysis
  analysis_name_2: "" # if files have different names override them (--kwargs)

  # Global plot aesthetics settings
  use_seaborn: True
  matplotlib_settings:
    # recommended colormaps: 'viridis' or 'cividis'
    cmap: "cividis"
    # recommended styles: 'tableau-colorblind10' or 'seaborn-colorblind'
    style: "seaborn-colorblind"
    rc: { "font_size": 8, "font_family": "Fira Sans" }
    scale: 1.5 # scale all plots by a factor
  seaborn_settings:
    theme:
      style: "whitegrid"
      context: "talk"
    # override context and/or style
    rc_context: {}
    rc_style: { "xtick.bottom": True, "ytick.left": True }

  # Requirements data
  load_requirements: True
  requirements_input_directory: ""

  # CTAMARS reference data
  # available at https://forge.in2p3.fr/projects/step-by-step-reference-mars-analysis/wiki
  load_CTAMARS: False
  # this is a setup *required* to run the notebooks smoothly!
  input_data_CTAMARS:
    parent_directory: ""
    TRAINING/DL1: "TRAINING/DL1"
    TRAINING/DL2: "TRAINING/DL2"
    DL2: "" # not available
    DL3:
      indir: "DL3"
      infile: ""
    label: "CTAMARS"

  # EVENTDISPLAY reference data (only ROOT format, for the moment)
  # available from https://forge.in2p3.fr/projects/cta_analysis-and-simulations/wiki#Instrument-Response-Functions
  load_EventDisplay: True
  input_data_EventDisplay:
    input_directory:
    input_file:
    label: "EventDisplay"

  # Input data
  input_filenames:
    # The simtel file is supposed to be used as a test run
    # WARNING: CTAMARS comparison requires a specific simtel file, see notebook.
    simtel: "" # (only) this is meant to be a full path
    # This is data produced with protopipe
    # These files are pre-defined so you shouldn't need to edit them
    TRAINING_energy_gamma: "TRAINING_energy_tail_gamma_merged.h5"
    TRAINING_classification_gamma: "TRAINING_classification_tail_gamma_merged.h5"
    TRAINING_classification_proton: "TRAINING_classification_tail_proton_merged.h5"
    DL2_gamma: "DL2_tail_gamma_merged.h5"
    DL2_proton: "DL2_energy_tail_gamma_merged.h5"
    DL2_electron: "DL2_energy_tail_gamma_merged.h5"
    # The DL3 filename depends on the simulation and analysis settings
    # Defined by editing performance.yaml
    DL3: ""

  model_configuration_filenames:
    energy: "RandomForestRegressor.yaml"
    classification: "RandomForestClassifier.yaml"

  # This MUST be data produced with ctapipe-process
  # with the JSON files available from protopipe or custom ones
  input_filenames_ctapipe:
    DL1a_gamma: "events_protopipe_CTAMARS_calibration_1stPass.dl1.h5"
    DL1a_gamma_2ndPass: "events_protopipe_CTAMARS_calibration_2ndPass.dl1.h5"
