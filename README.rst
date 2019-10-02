protopipe
=========

.. image:: https://api.codacy.com/project/badge/Grade/32f2fb2df3154fa1838c765d4f9110ba
    :target: https://www.codacy.com/app/karl.kosack/protopipe?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cta-observatory/protopipe&amp;utm_campaign=Badge_Grade

A pipeline prototype for the Cherenkov Telescope Array. To build the docs:

.. code-block:: bash

	>$ cd docs
	>$ make html
	>$ display _build/html/index.html


Requirements:
-------------
- [ctapipe=0.7](https://github.com/cta-observatory/ctapipe)
- [gammapy=0.8](https://github.com/gammapy/)

Basic use:
----------

The following instructions refer to local use of protopipe, which doesn't
involve the use of the DIRAC computing grid (ignore the grid.yaml configuration file).

A typical workflow consists in:

1) create an analysis parent folder with the auxiliary script create_dir_structure.py
2) prepare the configuration files
   1) copy the example YAML configuration files in the relative subfolders
   1) edit them for the particular needs of your analysis
3) build a model for energy estimation
   1) Create tables for gamma-rays using write_dl1.py and analysis.yaml
   2) Merge them with the auxiliary script merge.sh
   3) create the model with build_model.py and regressor.yaml
   4) and check it's performance with model_diagnostic.py and regressor.yaml
1) build a model for gamma/hadron separation
   1) same process as for the energy estimation but using classifier.yaml
      1) set 'estimate_energy' to True when using write_dl1.py so the reconstructed energy can be estimated and further used as a discriminant parameter.
1) produce DL2-level data
   1) create tables for gamma-rays, protons and electrons using write_dl2.py and analysis.yaml
   2) merge them with the auxiliary script merge.sh
1) Estimate the final performance with make_performance.py and performance.yaml
