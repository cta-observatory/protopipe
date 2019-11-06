protopipe
=========

![Code Quality](https://api.codacy.com/project/badge/Grade/32f2fb2df3154fa1838c765d4f9110ba)

A pipeline prototype for the Cherenkov Telescope Array.

Installation
------------

Get the source code and create the required basic conda environment:

          git clone https://github.com/cta-observatory/protopipe.git
          cd protopipe
          conda env create -f protopipe_environment.yml
          conda activate protopipe
          
In case you have already an environment 'myenv' set up for that kind of development, use:

          conda env update -n myenv -f protopipe_environment.yml
          
This environment contains the bare minimum in order to run the scripts and build the documentation.

It doesn't take into account any additional tool you could use later on (it is suggested to install _ipython_, _jupyter_ and _vitables_, especially if you want to contribute to the code).

Next you need to install _protopipe_ itself (_protopipe_ is the folder where the code was expanded. The name differs if you downloaded a released tagged version, it can be _protopipe-0.2_ for instance) :

          cd protopipe
          python setup.py develop

This will let you make changes to your local git repository without the need to update your environment every time.
In the abscence of a _conda_ or _pip_ installation made avalaible, to install a released version (no further development), use :

          python setup.py install

Remember that the environment needs to be activated in order for _protopipe_ to work.
This procedure has been successfully tested on macOS (10.10.5 & 10.14.6) and on Scientific Linux 7.

Building the documentation
--------------------------

From the main folder go down to the documentation repository and create the documentation :

          cd docs
          make html

You will probably get some harmless warnings.
The initial page is stored in _ _build/html/index.html_, which you can open using your favorite internet browser.

Test if it works
-----------------
Before starting to use protopipe, be sure to be inside the relevant environment (e.g. """conda activate protopipe""").
Get a _simtel_ Monte Carlo file (mydata.gz) obtained with _prod3b_ anth the La Palma site and produce a simple DL1 output (It will not work for a paranal file):

          cd protopipe/scripts
          python write_dl1.py -h
          
To be completed (it crashes so far)...

Analysis chain general description
----------------------------------
The following instructions refer to local use of protopipe, which doesn't involve the use of the DIRAC computing grid (you can ignore the grid.yaml configuration file).
Typical analysis steps are the following :

1. **create an analysis parent folder** with the auxiliary script _create_dir_structure.py_
2. **prepare the configuration files**
    1. copy the example YAML configuration files in the relative subfolders
    2. edit them for the particular needs of your analysis
3. **build a model for energy estimation**
    1. Create tables for gamma-rays using _write_dl1.py_ and _analysis.yaml_
    2. Merge them with the auxiliary script _merge.sh_
    3. create the model with _build_model.py_ and _regressor.yaml_
    4. check it's performance with _model_diagnostic.py_ and _regressor.yaml_
4. **build a model for gamma/hadron separation**
    1. set 'estimate_energy' to True when using _write_dl1.py_ so the reconstructed energy can be estimated and further used as a discriminant parameter.
    2. same process as for the energy estimation but using _classifier.yaml_
5. **produce DL2-level data**
    1. create tables for gamma-rays, protons and electrons using _write_dl2.py_ and _analysis.yaml_
    2. merge them with the auxiliary script _merge.sh_
6. **Estimate the final performance (IRFs)** with _make_performance.py_ and _performance.yaml_

_**Note:**_ DL1/DL2 scripts take as input only 1 file at the time.

Instructions for developers:
----------------------------

1. Fork the master _protopipe_ remote repository as explained [here](https://help.github.com/en/articles/fork-a-repo)
2. Follow the installation instructions above, but using __your__ remote repository (we'll call __origin__ yours and __upstream__ the official one)
3. if you want to develop something new:
    1. update __local__ _master_ branch (`git pull upstream master`)
    3. create a new branch from your __local__ _master_ branch
    4. develop inside it
    5. push it to __origin__
    6. continue to develop and push until you feel ready
4. start a __pull request__ from __origin/your_branch__ to __upstream/master__
    1. wait for an outcome
    2. if necessary, you can update or fix things in your branch because now everything is traced (__local/your_branch__ --> __origin/your_branch__ --> __pull request__)

_**Note:**_ if your developments take a relatively long time,

1. update periodically your __local__ _master_ branch,
2. if updates have been made, go to your __local__ _development_ branch (`git checkout your_branch`)
3. if there are no conflicts, move the beginning of your branch at the end of the updated master (`git rebase master`)
4. push your branch to your remote
