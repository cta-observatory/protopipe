.. _use-grid:

Large scale productions
=======================

.. note::
  Before launching any job on the grid, it is assumed that you have followed
  the required installation instructions (:ref:`install-grid`).
  Please, check that your shell can recognize the environment variables
  $PROTOPIPE and $GRID, and that such variables point to the correct paths.


The following instructions are built on the workflow described in (:ref:`usage`)
, but they are executed in the grid environment.

1. Setup the analysis as explained in :ref:`usage`, in the shared ``data``
   folder, so you can access it from outside

2. Create the energy estimator

* trained the DL1 data that will be used to build the model
* ``python $GRID/submit_jobs_new_scheme.py --config_file=grid.yaml --output_type=DL1``
* edit and execute ``$GRID/download_and_merge.sh`` in order to download such data
  from the grid path defined in ``grid.yaml`` and merge it into a table hosted
  in the respective path of your analysis folder
* from outside of the grid environment build the model locally using protopipe
* ``python ./build_model.py --config_file=regressor.cfg --max_events=200000``
* from there you can also diagnose your models
* python ./model_diagnostic.py --config_file=regressor.cfg
* return to the grid environment to edit and execute ``./upload_models.sh``

2. Create the particle classifier

* set ``estimate_energy`` to ``True`` in order for the reconstructed energy to
  be estimated and further used as a discriminant parameters.
  In addition, this flag also indicates that the file lists should be taken in
  the ``GammaHadronClassifier`` section.
* next steps are analog to the previous section

3. Create the DL2 dataset

* ``python submit_jobs_new_scheme.py --config_file=grid.cfg --output_type=DL2``
* ``$GRID/download_and_merge.sh``
* you can exit the grid environment now

4. Estimate the performance

* ``$PROTOPIPE/protopipe/aux/scripts/multiple_performances.sh``
* the ``performance`` subfolder in your analysis parent folder should now
  contain  set of 4 folders, each containing the respective IRF information

  .. note::
    The notebooks/benchmarks required to visualize this data will be added soon.
