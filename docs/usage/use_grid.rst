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
  
  It is here assumed that you are working on a DIRAC-based grid environment.

  You will work with two different virtual environments:

  - protopipe (Python >=3.5, conda environment)
  - GRID interface (Python 2.7, inside the container).

  Open 1 tab for each of these environments on you terminal so you can work seamlessly between the 2.

  To monitor the jobs you can use the
  `DIRAC Web Interface <https://ccdcta-web.in2p3.fr/DIRAC/?view=tabs&theme=Crisp&url_state=1|*DIRAC.JobMonitor.classes.JobMonitor:,>`_

1. **Setup analysis** (GRID enviroment)

  After having entered the container, and assuming your username is _johndoe_
  with an output directory called e.g. _cta_analyses_
  
  | ``python $GRID_INTERFACE/create_analysis_tree.py --analysis_name myAnalysis``
  | ``--GRID-is-DIRAC --GRID-home /vo.cta.in2p3.fr/user/j/johndoe --GRID-path-from-home cta_analyses``

  This command will store and partially edit for you all the necessary
  configuration files under the ``configs`` folder as well as the operational
  scripts to download and upload data and model files under ``data`` and
  ``estimators`` respectively.
  
  Throughout the following instructions ``$ANALYSIS`` will be a label for the analysis
  name.

  .. figure:: ./AnalysisTree.png
    :width: 250
    :alt: Directory tree of a full analysis performed with protopipe.

2. **Obtain training data for energy estimation** (GRID enviroment)

  1. edit ``grid.yaml`` to use gammas without energy estimation
  2. ``python $GRID_INTERFACE/submit_jobs.py --analysis_name=$ANALYSIS --output_type=TRAINING``
  3. edit and execute ``$ANALYSIS/data/download_and_merge.sh`` once the files are ready

3. **Build the model for energy estimation** (both enviroments)

  1. switch to the ``protopipe environment``
  2. edit the configuration file of your model of choice
  3. use ``protopipe-MODEL`` with this configuration file
  4. use ``protopipe-BENCHMARK`` to check the performance of the generated models
  5. return to the ``GRID environment`` to edit and execute ``upload_models.sh`` from the estimators folder

4. **Obtain training data for particle classification** (GRID enviroment)

  1. edit ``grid.yaml`` to use gammas **with** energy estimation
  2. ``python $GRID_INTERFACE/submit_jobs.py --analysis_name=$ANALYSIS --output_type=TRAINING``
  3. edit and execute ``$ANALYSIS/data/download_and_merge.sh`` once the files are ready
  4. repeat the first 3 points for protons
  5. use ``protopipe-BENCHMARK`` to check the estimated energies

4. **Build a model for particle classification** (both enviroments)

  1. switch to the ``protopipe environment``
  2. edit ``RandomForestClassifier.yaml``
  3. use ``protopipe-MODEL`` with this configuration file
  4. use ``protopipe-BENCHMARK`` to check the performance of the generated models
  5. return to the ``GRID environment`` to edit and execute ``upload_models.sh`` from the ``estimators`` folder

5. **Get DL2 data** (GRID enviroment)

Execute points 1 and 2 for gammas, protons, and electrons separately.

  1. ``python $GRID_INTERFACE/submit_jobs.py --analysis_name=$ANALYSIS --output_type=DL2``
  2. edit and execute ``download_and_merge.sh``
  3. use ``protopipe-BENCHMARK`` to check the quality of the generated DL2 data

6. **Estimate the performance** (protopipe enviroment)

  1. edit ``performance.yaml``
  2. launch the performance script with this configuration file and an observation time
  3. use ``protopipe-BENCHMARK`` to check the quality of the generated DL3 data


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
