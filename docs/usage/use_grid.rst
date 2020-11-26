.. _use-grid:

Large scale productions
=======================

Requirements
------------

* a working installation of both protopipe (:ref:`install`) and its interface on the gris (:ref:`install-grid`),
* be accustomed with the basic pipeline workflow (:ref:`use-pipeline`).

Usage
-----

.. note::

  You will work with two different environments to deal with protopipe
  (Python >=3.5, conda environment)
  and the grid utilities (Python 2.7, built-in environment).

1. Setup the analysis (step 1 of :ref:`use-pipeline`) in the shared ``data``
   folder, so you can access it from outside

2. Edit the configuration files according to your GRID user information and
   your analysis settings

3. Build a model for energy estimation

  * train the data that will be used to build the model

    - ``python $GRID/submit_jobs_new_scheme.py --config_file=grid.yaml --output_type=DL1``,
    - copy, edit and execute ``$GRID/download_and_merge.sh`` in order to download such data
      from the grid path defined in ``grid.yaml`` and merge it into a table hosted
      in the appropriate path of your analysis folder (``[...]/data/DL1/for_energy_estimation``)

  * build the model

    - open a new tab in your terminal and go to the model folder, activating the protopipe environment,
    - ``python ./build_model.py --config_file=regressor.yaml --max_events=200000``

  * operate some diagnostics

    - ``python ./model_diagnostic.py --config_file=regressor.yaml``
    - associate benchmark notebooks to be added soon

  * upload the model on the grid

    - return to the grid environment to edit and execute ``$GRID/upload_models.sh``

4. Build a model for particle classification

  * edit ``grid.yaml`` by setting ``estimate_energy`` to ``True`` in order for the reconstructed energy to
    be estimated and further used as a discriminant parameters.
    In addition, this flag also indicates that the file lists should be taken in
    the ``GammaHadronClassifier`` section.
  * next steps are analog to Step 3

5. Create the DL2 dataset

  * ``python $GRID/submit_jobs_new_scheme.py --config_file=grid.yaml --output_type=DL2``
  * ``. download_and_merge.sh``
  * you can now exit the grid environment

6. Estimate the performance

  * ``. $PROTOPIPE/protopipe/aux/scripts/multiple_performances.sh``
  * the ``performance`` subfolder in your analysis parent folder should now
    contain a set of 4 folders, each containing the respective IRF information
  * associate benchmark notebooks to be added soon

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
