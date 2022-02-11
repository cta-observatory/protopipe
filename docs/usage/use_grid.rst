.. _use-grid:

Large scale analyses
====================

This page is meant to be also a tutorial on how to perform large scale analyses with _protopipe_.

Requirements
------------

* protopipe (:ref:`install`)
* GRID interface (:ref:`install-grid`),
* be accustomed with the basic pipeline workflow (:ref:`use-pipeline`).

.. figure:: ./GRID_workflow.png
  :width: 800
  :alt: Standard analysis workflow with protopipe

  Standard analysis workflow with protopipe.

Usage
-----

Preliminary operations
^^^^^^^^^^^^^^^^^^^^^^

1. Get a proxy with ``dirac-proxy-init``
2. Choose the simulation production that you want to work on (this is reported e.g. in the CTA wiki)
3. use ``cta-prod-dump-dataset`` to obtain the original lists of simtel files for each particle type
4. use ``protopipe-SPLIT_DATASET`` to split the original lists depending on the chosen analysis workflow

.. warning::

  For the moment, only the standard analysis workflow (stored under ``protopipe_grid_interface/aux``) has been tested.

.. note::

  Make sure you are using a `protopipe-CTADIRAC`-based environment, otherwise you won't
  be able to interface with DIRAC.

  To monitor the jobs you can use the 
  `DIRAC Web Interface <https://ccdcta-web.in2p3.fr/DIRAC/?view=tabs&theme=Crisp&url_state=1|*DIRAC.JobMonitor.classes.JobMonitor:,>`_

Start your analysis
^^^^^^^^^^^^^^^^^^^

1. **Setup analysis**

  Use ``protopipe-CREATE_ANALYSIS`` to create a directory tree, depending on your setup.

  The script will store and partially edit for you all available
  configuration files under the ``configs`` folder.
  It will also create an ``analysis_metadata.yaml`` file which will store the
  basic information regarding your analysis on the grid and on your machine.

  .. note::

    To reproduce or access the analysis data of someone else on DIRAC it will be sufficient
    to modify the metadata key ``analyses_directory`` referred to your local path.

  .. figure:: ./AnalysisTree.png
    :width: 250
    :alt: Directory tree of a full analysis performed with protopipe.

2. **Obtain training data for energy estimation**

  1. edit ``grid.yaml`` to use gammas without energy estimation
  2. ``protopipe-SUBMIT_JOBS --analysis_path=[...]/test_analysis --output_type=TRAINING ....``
  3. once the jobs have concluded and the files are ready you can use ``protopipe-DOWNLOAD_AND_MERGE``
  4. (development users) use the proper benchmarking notebooks under ``docs/contribute/benchmarks`` to check the properties of the data sample you obtained

3. **Build the model for energy estimation**

  1. edit the configuration file of your model of choice
  2. use ``protopipe-MODEL`` with this configuration file
  3. (development users) use the proper benchmarking notebooks under ``docs/contribute/benchmarks`` to check the performance of the generated models
  4. use ``protopipe-UPLOAD_MODELS`` to upload models and configuration file to your analysis directory on the DIRAC File Catalog

4. **Obtain training data for particle classification**

  1. edit ``grid.yaml`` to use gammas **with** energy estimation
  2. ``protopipe-SUBMIT_JOBS --analysis_path=[...]/test_analysis --output_type=TRAINING ....``
  3. once the jobs have concluded and the files are ready you can use ``protopipe-DOWNLOAD_AND_MERGE``
  4. repeat the first 3 points for protons
  5. (development users) use the proper benchmarking notebooks under ``docs/contribute/benchmarks`` to check the quality of energy estimation on this data sample

4. **Build a model for particle classification**

  1. edit ``RandomForestClassifier.yaml``
  2. use ``protopipe-MODEL`` with this configuration file
  3. (development users) use the proper benchmarking notebooks under ``docs/contribute/benchmarks`` to check the performance of the generated models
  4. use ``protopipe-UPLOAD_MODELS`` to upload models and configuration file to your analysis directory on the DIRAC File Catalog

5. **Get DL2 data**

Execute points 1 and 2 for gammas, protons, and electrons separately.

  1. ``protopipe-SUBMIT_JOBS --analysis_path=[...]/test_analysis --output_type=DL2 ....``
  2. once the jobs have concluded and the files are ready you can use ``protopipe-DOWNLOAD_AND_MERGE``
  3. (development users) use the proper benchmarking notebooks under ``docs/contribute/benchmarks`` to check the quality of the generated DL2 data

6. **Estimate the performance** (protopipe enviroment)

  1. edit ``performance.yaml``
  2. ``protopipe-DL3-EventDisplay`` with this configuration file
  3. (development users) use the proper benchmarking notebooks under ``docs/contribute/benchmarks`` to check the quality of the generated DL3 data


Troubleshooting
---------------

Issues with the login
^^^^^^^^^^^^^^^^^^^^^

**After issuing the command ``dirac-proxy-init`` I get the message
"Your host clock seems to be off by more than a minute! Thats not good.
We'll generate the proxy but please fix your system time" (or similar)**

From within the Vagrant Box environment execute these commands:

- ``systemctl status systemd-timesyncd.service``
- ``sudo systemctl restart systemd-timesyncd.service``
- ``timedatectl``

Check that,

- ``System clock synchronized: yes``
- ``systemd-timesyncd.service active: yes``

**After issuing the command ``dirac-proxy-init`` and typing my certificate
password the process start pending and gets stuck**

One possible reason might be related to your network security settings.
Some networks might require to add the option ``-L`` to ``dirac-proxy-init``.

Issues with the download
^^^^^^^^^^^^^^^^^^^^^^^^

**After correctly editing and launching the ``download_and_merge.sh`` script
I get "UTC Framework/API ERROR: Failures occurred during rm.getFile"**

Something went wrong during the download phase, either because of your network
connection (check for possible instabilities) or because of a problem
on the server side (in which case the solution is out of your control).

First let the process finish and eliminate the incomplete merged file, then
the recommended approach is to use the DIRAC's command,

``dirac-dms-directory-sync source destination``

where ``source`` is the LFN on DIRAC's FileCatalog and ``destination`` is the
target folder under you analysis directory tree.

If this doesn't work, a more manual approach is:

- go to the GRID, copy the list of files and dump it into e.g. ``grid.list``,
- do the same with the local files into e.g. ``local.list``,
- do ``diff <(sort local.list) <(sort grid.list)``,
- download the missing files with ``dirac-dms-get-file``,
- modify (temporarily) ``download_and_merge.sh`` by commenting the
  download line and execute it so you just merge them.

Issues with the job submission
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**I get an error like "Data too long for column 'JobName' at row 1" or similar**

The job name is too long, try to modify it temporarily by editing submit_jobs.py.
There will be soon an option to modify it at launch time.

**I get an error which starts with 'FileCatalog._getEligibleCatalogs: Failed to get file catalog configuration. Path /Resources/FileCatalogs does not exist or it's not a section'**

This is a Configuration System error which is not fully debugged yet.
Check that your dirac.cfg file is correctly edited.
In some cases the interface code will re-try to issue some commands in case this happens.