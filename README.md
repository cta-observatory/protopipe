protopipe
=========

.. image:: https://api.codacy.com/project/badge/Grade/32f2fb2df3154fa1838c765d4f9110ba
    :target: https://www.codacy.com/app/karl.kosack/protopipe?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cta-observatory/protopipe&amp;utm_campaign=Badge_Grade

A pipeline prototype for the Cherenkov Telescope Array.

Installation:
-------------

You can create directly a conda environment by issuing the following command,

          conda env create -f protopipe_environment.yml

This environment contains the bare minimum in order to run the scripts and build the documentation. 

It doesn't take into account any additional tool you could use later on (even though it is suggested to install 'ipython', 'jupyter' and 'vitables' if you want to help us develop).

The module named 'pywi-cta' has to be installed from the git repository https://github.com/HealthyPear/pywi-cta.git using the 'setup.py' file,

          python setup.py install

Finally, you need to launch the installation of protopipe itself from using a similar command,

          python setup.py develop

which will let you play within the git repository and use protopipe in a continuous way.

Building the documentation:
---------------------------

.. code-block:: bash

	>$ cd docs
	>$ make html
	>$ display _build/html/index.html (or 'open' if you work on macos)

You will probably get some harmless warnings.

Basic use:
----------

The following instructions refer to local use of protopipe, which doesn't
involve the use of the DIRAC computing grid (ignore the grid.yaml configuration file).

Before starting to use protopipe, you have to place your working environment
inside the one you created at the beggining:

          conda activate protopipe

Then, a typical workflow consists in:

 1. create an analysis parent folder with the auxiliary script create_dir_structure.py
 2. prepare the configuration files
    1. copy the example YAML configuration files in the relative subfolders
    2. edit them for the particular needs of your analysis
 3. build a model for energy estimation
    1. Create tables for gamma-rays using write_dl1.py and analysis.yaml
    2. Merge them with the auxiliary script merge.sh
    3. create the model with build_model.py and regressor.yaml
    4. check it's performance with model_diagnostic.py and regressor.yaml
 4. build a model for gamma/hadron separation
    1. set 'estimate_energy' to True when using write_dl1.py so the reconstructed energy can be estimated and further used as a discriminant parameter.
    2. same process as for the energy estimation but using classifier.yaml
 5. produce DL2-level data
    1. create tables for gamma-rays, protons and electrons using write_dl2.py and analysis.yaml
    2. merge them with the auxiliary script merge.sh
 6. Estimate the final performance (IRFs) with make_performance.py and performance.yaml
