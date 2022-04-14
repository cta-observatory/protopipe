.. _dirac:

Interface to the DIRAC grid
===========================

This part of the documentation covers the set of commands provided by the
`protopipe-grid-interface <https://github.com/HealthyPear/protopipe-grid-interface>`__

.. note::

   This package is currently stand-alone.  
   It is planned to merge it into *protopipe* as a new module `protopipe.dirac`.

The most important commands provided by this interface are listed here,

- :ref:`new`
- :ref:`split`
- :ref:`submit`
- :ref:`download-and-merge`
- :ref:`upload-models`

There are secondary scripts there are mainly used for debugging.
Please refer to their help (`command_name -h`) in case you need to use them.

.. _new:

Create a new analysis
---------------------

``protopipe-CREATE_ANALYSIS`` assumes you work from a common parent directory named `shared_folder` (required) which will host both
the `analyses` directory and a `productions` directory.

.. note::

   The name of the parent directory could change, it was named like this due to the fact
   that previous versions of the interface required a containerized solution and the naming
   was clearer in that context.

.. code-block::

    usage: protopipe-CREATE_ANALYSIS [-h] --analysis_name ANALYSIS_NAME [--analysis_directory_tree ANALYSIS_DIRECTORY_TREE] [--log_file LOG_FILE] [--output_path OUTPUT_PATH] [--GRID-is-DIRAC]
                                    [--GRID-home GRID_HOME] [--GRID-path-from-home GRID_PATH_FROM_HOME] [--overwrite-analysis]

    Create a directory structure for a CTA data analysis using the protopipe prototype pipeline.

        WARNING: check that the version of protopipe is the one you intend to use!



    optional arguments:
    -h, --help            show this help message and exit
    --analysis_name ANALYSIS_NAME
                            Name of the analysis
    --analysis_directory_tree ANALYSIS_DIRECTORY_TREE
                            Analysis workflow YAML file (default: see protopipe_grid_interface.aux)
    --log_file LOG_FILE   Override log file path
                                            (default: analysis.log in analysis folder)
    --output_path OUTPUT_PATH
                            Full path where the 'shared_folder' should be created (or where it already exists)
    --GRID-is-DIRAC       The grid on which to run the analysis is the DIRAC grid.
    --GRID-home GRID_HOME
                            Path of the user's home on grid (if DIRAC, /vo.cta.in2p3.fr/user/x/xxx).
    --GRID-path-from-home GRID_PATH_FROM_HOME
                            Path of the analysis on the DIRAC grid. Defaults for empty string (user home)
    --overwrite-analysis  Overwrite analysis folder (WARNING: you could loose data!)


.. important::

    ``protopipe-CREATE_ANALYSIS`` creates an ``analysis_metadata.yaml`` file with the provided input information.
    Many commands benefit from using this file to retrieve such information at runtime so you don't have to
    input it again.
    It can be also useful to retrieve data and configuration files from the DIRAC file catalog if you want to
    work on someone's analysis.

    It will also create an ``analysis.log`` file which will be filled anytime you use the main commands,
    unless another location is specified.

.. _split:

Split datasets
--------------

After you created a new analysis, head over to the list of available simulation datasets
(either from the CTA wiki or using CTADIRAC tools) and retrieve the original lists of
simtel files per particle type.

With ``protopipe-SPLIT_DATASET`` you can split these lists in sub-datasets to be stored under the
`data/simtel` folder within your analysis.

.. code-block::

    usage: protopipe-SPLIT_DATASET [-h] [--metadata METADATA | --output_path OUTPUT_PATH] [--input_gammas INPUT_GAMMAS] [--input_protons INPUT_PROTONS] [--input_electrons INPUT_ELECTRONS]
                                [--split_gammas SPLIT_GAMMAS [SPLIT_GAMMAS ...]] [--split_protons SPLIT_PROTONS [SPLIT_PROTONS ...]] [--split_electrons SPLIT_ELECTRONS [SPLIT_ELECTRONS ...]]
                                [--log_file LOG_FILE]

    Split a simulation dataset.

        Requirement:
        - list files should have *.list extension.

        Default analysis workflow (see protopipe/aux/standard_analysis_workflow.yaml):
        - a training sample for energy made of gammas,
        - a training sample for particle classification made of gammas and protons,
        - a performance sample made of gammas, protons, and electrons.



    optional arguments:
    -h, --help            show this help message and exit
    --metadata METADATA   Analysis metadata file produced at creation
                                    (recommended).
    --output_path OUTPUT_PATH
                            Specifiy an output directory
    --input_gammas INPUT_GAMMAS
                            Full path of the original list of gammas.
    --input_protons INPUT_PROTONS
                            Full path of the original list of protons.
    --input_electrons INPUT_ELECTRONS
                            Full path of the original list of electrons.
    --split_gammas SPLIT_GAMMAS [SPLIT_GAMMAS ...]
                            List of percentage values in which to split the gammas. Default is [10,10,80]
    --split_protons SPLIT_PROTONS [SPLIT_PROTONS ...]
                            List of 3 percentage values in which to split the protons. Default is [40,60]
    --split_electrons SPLIT_ELECTRONS [SPLIT_ELECTRONS ...]
                            List of percentage values in which to split the electrons. Default is [100]
    --log_file LOG_FILE   Override log file path
                                            (ignored when using metadata)

.. _submit:

Submit a set of jobs
--------------------

``protopipe-SUBMIT-JOBS`` needs always a valid `grid.yaml` configuration file.

.. code-block::

    Usage:
    protopipe-SUBMIT-JOBS [options] ...

    General options:
    -o  --option <value>         : Option=value to add
    -s  --section <value>        : Set base section for relative parsed options
    -c  --cert <value>           : Use server certificate to connect to Core Services
    -d  --debug                  : Set debug mode (-ddd is extra debug)
    -   --cfg=                   : Load additional config file
    -   --autoreload             : Automatically restart if there's any change in the module
    -   --license                : Show DIRAC's LICENSE
    -h  --help                   : Shows this help

    Options:
    -   --analysis_path=         : Full path to the analysis folder
    -   --output_type=           : Output data type (TRAINING or DL2)
    -   --max_events=            : Max number of events to be processed (optional, int)
    -   --upload_analysis_cfg=   : If True (default), upload analysis configuration file
    -   --dry=                   : If True do not submit job (default: False)
    -   --test=                  : If True submit only one job (default: False)
    -   --save_images=           : If True save images together with parameters (default: False)
    -   --debug_script=          : If True save debug information during execution of the script (default: False)
    -   --DataReprocessing=      : If True reprocess data from one site to another (default: False)
    -   --tag=                   : Used only if DataReprocessing is True; only sites tagged with tag will be considered (default: None)
    -   --log_file=              : Override log file path (default: analysis.log in analysis folder)

.. _download-and-merge:

Download data and merge it
--------------------------

``protopipe-DOWNLOAD_AND_MERGE`` allows to download data serially, but also in ``rsync``-style
(this second option is used as a backup crosscheck automatically in fact, to overcome any
network malfunction) and merge it in a single HDF5 file.

.. warning::
    Currently the merging is done locally, so both single files and merged file will be stored
    locally resulting in a two-fold amount of disk-space usage.
    This is of course not ideal: data should be merged on the grid.

.. code-block::

    usage: protopipe-DOWNLOAD_AND_MERGE [-h] [--metadata METADATA] [--disable_download] [--disable_sync] [--disable_merge] [--indir INDIR] [--outdir OUTDIR] --data_type
                                        {TRAINING/for_energy_estimation,TRAINING/for_particle_classification,DL2} --particle_types [{gamma,proton,electron} [{gamma,proton,electron} ...]] [--n_jobs N_JOBS]
                                        [--cleaning_mode CLEANING_MODE] [--GRID-home GRID_HOME] [--GRID-path-from-home GRID_PATH_FROM_HOME] [--analysis_name ANALYSIS_NAME] [--local_path LOCAL_PATH]
                                        [--log_file LOG_FILE]

    Download and merge data from the DIRAC grid.

        The default behaviour calls an rsync-like command after the normal download as
        an additional check.

        This script can be used separately, or in association with an analysis workflow.
        In the second case the recommended usage is via the metadata file produced at creation.


    optional arguments:
    -h, --help            show this help message and exit
    --metadata METADATA   Path to the metadata file produced at analysis creation
                                    (recommended - if None, specify necessary information).
    --disable_download    Do not download files serially
    --disable_sync        Do not syncronyze folders after serial download
    --disable_merge       Do not merge files at the end
    --indir INDIR         Override input directory
    --outdir OUTDIR       Override output directory
    --data_type {TRAINING/for_energy_estimation,TRAINING/for_particle_classification,DL2}
                            Type of data to download and merge
    --particle_types [{gamma,proton,electron} [{gamma,proton,electron} ...]]
                            One of more particle type to download and merge
    --n_jobs N_JOBS       Number of parallel jobs for directory syncing (default: 4)
    --cleaning_mode CLEANING_MODE
                            Deprecated argument
    --GRID-home GRID_HOME
                            Path of the user's home on DIRAC grid (/vo.cta.in2p3.fr/user/x/xxx)
                                            (recommended: use metadata file)
    --GRID-path-from-home GRID_PATH_FROM_HOME
                            optional additional path from user's home in DIRAC (recommended: use metadata file)
    --analysis_name ANALYSIS_NAME
                            Name of the analysis (recommended: use metadata file)
    --local_path LOCAL_PATH
                            Path where shared_folder is located (recommended: use metadata file)
    --log_file LOG_FILE   Override log file path
                                            (default: analysis.log in analysis folder)

.. _upload-models:

Upload models
-------------

``protopipe-UPLOAD_MODELS`` allows you to upload both model files and their configuration files
to the DIRAC file catalog.
It allows to define a list of storage elements (SEs) where to store this data.
It will always try to upload to CC-IN3P3 first.

.. code-block::

    usage: protopipe-UPLOAD_MODELS [-h] [--metadata METADATA] --cameras CAMERAS [CAMERAS ...] --model_type {regressor,classifier} --model_name {RandomForestRegressor,AdaBoostRegressor,RandomForestClassifier}
                                [--cleaning_mode CLEANING_MODE] [--GRID-home GRID_HOME] [--GRID-path-from-home GRID_PATH_FROM_HOME] [--list-of-SEs [LIST_OF_SES [LIST_OF_SES ...]]] [--analysis_name ANALYSIS_NAME]
                                [--local_path LOCAL_PATH] [--log_file LOG_FILE]

    Upload models produced with protopipe to the Dirac grid.

        Files will be uploaded at least on CC-IN2P3-USER.
        You can use `cta-prod-show-dataset YOUR_DATASET_NAME --SEUsage` to know
        on which Dirac Storage Elements to replicate you models.
        Note: you will see *-Disk entries, but you need to replicate using *-USER entries.
        The default behaviour is replicate them to "DESY-ZN-USER", "CNAF-USER", "CEA-USER".
        Replication is optional.


    optional arguments:
    -h, --help            show this help message and exit
    --metadata METADATA   Path to the metadata file produced at analysis creation
                                    (recommended - if None, specify necessary information).
    --cameras CAMERAS [CAMERAS ...]
                            List of cameras to consider
    --model_type {regressor,classifier}
                            Type of model to upload
    --model_name {RandomForestRegressor,AdaBoostRegressor,RandomForestClassifier}
                            Type of model to upload
    --cleaning_mode CLEANING_MODE
                            Deprecated argument
    --GRID-home GRID_HOME
                            Path of the user's home on grid: /vo.cta.in2p3.fr/user/x/xxx (recommended: use metadata file).
    --GRID-path-from-home GRID_PATH_FROM_HOME
                            Analysis path on DIRAC grid (defaults to user's home; recommended: use metadata file)
    --list-of-SEs [LIST_OF_SES [LIST_OF_SES ...]]
                            List of DIRAC Storage Elements which will host the uploaded models
    --analysis_name ANALYSIS_NAME
                            Name of the analysis (recommended: use metadata file)
    --local_path LOCAL_PATH
                            Path where shared_folder is located (recommended: use metadata file)
    --log_file LOG_FILE   Override log file path
                                            (default: analysis.log in analysis folder)
