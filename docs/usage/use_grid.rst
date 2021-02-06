.. _use-grid:

Large scale analyses
====================

Requirements
------------

* protopipe (:ref:`install`)
* GRID interface (:ref:`install-grid`),
* be accustomed with the basic pipeline workflow (:ref:`use-pipeline`).

.. figure:: ./GRID_workflow.png
  :width: 800
  :alt: Workflow of a full analysis on the GRID with protopipe

  Workflow of a full analysis on the GRID with protopipe.

Usage
-----

.. note::

  You will work with two different virtual environments:

  - protopipe (Python >=3.5, conda environment)
  - GRID interface (Python 2.7, inside the container).

  Open 1 tab for each of these environments on you terminal so you can work seamlessly between the 2.

  To monitor the jobs you can use the
  `DIRAC Web Interface <https://ccdcta-web.in2p3.fr/DIRAC/?view=tabs&theme=Crisp&url_state=1|*DIRAC.JobMonitor.classes.JobMonitor:,>`_

1. **Setup analysis** (GRID enviroment)

  1. Enter the container
  2. ``python $GRID/create_analysis_tree.py --analysis_name myAnalysis``

  All configuration files for this analysis are stored under ``configs``.
  Throughout these instructions ``$ANALYSIS`` will be a label for the analysis
  path within or outside of the container.

  .. figure:: ./AnalysisTree.png
    :width: 250
    :alt: Directory tree of a full analysis performed with protopipe.

2. **Obtain training data for energy estimation** (GRID enviroment)

  1. edit ``grid.yaml`` to use gammas without energy estimation
  2. ``python $GRID/submit_jobs.py --config_file=grid.yaml --output_type=TRAINING``
  3. edit and execute ``$ANALYSIS/data/download_and_merge.sh`` once the files are ready

3. **Build the model for energy estimation** (both enviroments)

  1. switch to the ``protopipe environment``
  2. edit ``regressor.yaml``
  3. launch the ``build_model.py`` script of protopipe with this configuration file
  4. you can operate some diagnostics with ``model_diagnostic.py`` using the same configuration file
  5. diagnostic plots are stored in subfolders together with the model files
  6. return to the ``GRID environment`` to edit and execute ``upload_models.sh`` from the estimators folder

4. **Obtain training data for particle classification** (GRID enviroment)

  1. edit ``grid.yaml`` to use gammas **with** energy estimation
  2. ``python $GRID/submit_jobs.py --config_file=grid.yaml --output_type=TRAINING``
  3. edit and execute ``$ANALYSIS/data/download_and_merge.sh`` once the files are ready
  4. repeat the first 3 points for protons

4. **Build a model for particle classification** (both enviroments)

  1. switch to the ``protopipe environment``
  2. edit ``classifier.yaml``
  3. launch the ``build_model.py`` script of protopipe with this configuration file
  4. you can operate some diagnostics with ``model_diagnostic.py`` using the same configuration file
  5. diagnostic plots are stored in subfolders together with the model files
  6. return to the ``GRID environment`` to edit and execute ``upload_models.sh`` from the estimators folder

5. **Get DL2 data** (GRID enviroment)

Execute points 1 and 2 for gammas, protons, and electrons separately.

  1. ``python $GRID/submit_jobs.py --config_file=grid.yaml --output_type=DL2``
  2. edit and execute ``download_and_merge.sh``

6. **Estimate the performance** (protopipe enviroment)

  1. edit ``performance.yaml``
  2. launch the performance script with this configuration file and an observation time


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

The best approach is:

- let the process finish and eliminate the incomplete merged file,
- go to the GRID, copy the list of files and dump it into e.g. ``grid.list``,
- do the same with the local files into e.g. ``local.list``,
- do ``diff <(sort local.list) <(sort grid.list)``,
- download the missing files with ``dirac-dms-get-file``,
- modify (temporarily) ``download_and_merge.sh`` by commenting the
  download line and execute it so you just merge them.
