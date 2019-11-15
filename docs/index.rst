============================
A pipeline protopipe for CTA
============================

.. currentmodule:: protopipe

What is protopipe?
==================
`Protopipe` is a pipeline prototype for the `Cherenkov Telescope Array
<https://www.cta-observatory.org/>`_ (CTA) based on the `ctapipe
<https://cta-observatory.github.io/ctapipe/>`_ library.
The package is currently developed and tested at the CEA in the department
of astrophysics.

The pipeline provides scripts to:

 * Process simtelarray files and write DL1 or DL2 tables
 * Build regression or classification models with diagnostic plots
 * Estimate the best cutoffs which gives the minimal sensitivity
   reachable in a given amount of time
 * Produce instrument response functions (IRF), including sensitivity

In order to process a significant amount of events the use of the GRID is rapidly
mandatory. Some utility scripts to submits jobs on the GRID are provided on
the `GRID repository <https://drf-gitlab.cea.fr/CTA-Irfu/grid>`_.

.. warning::

  | For the moment *protopipe* supports (and is tested with) simtel Monte Carlo files obtained with prod3b with only LSTCam and NectarCam cameras.
  | Any other kind of camera could lead to a crash (see e.g. `this <https://github.com/cta-observatory/protopipe/issues/22>`_ open issue).
  | Note that some generic La Palma files can contain FlashCam cameras.

.. warning::
  This is not yet stable code, so expect large and rapid changes.

Installation
============

Requirements
------------

The only requirement is an Anaconda installation which supports Python 3.


.. Note::

  For a faster use, edit your preferred login script (e.g. ``.bashrc`` or
  ``.profile``) with a function that initializes the environment. The following
  is a minimal example using Bash.

  .. code-block:: bash

    function protopipe_init() {

        conda activate protopipe # Then activate the protopipe environment
        export PROTOPIPE=$WHEREISPROTOPIPE/protopipe # A shortcut to the scripts folder

    }

Instructions for basic users
----------------------------

If you are a basic user with no interest in developing *protopipe*, you can use
the latest released version that you can find
`here <https://github.com/cta-observatory/protopipe/releases>`__ as a compressed archive.

Steps for installation:

  1. uncompress the file which is always called *protopipe-X.Y.Z* depending on version,
  2. enter the folder ``cd protopipe-X.Y.Z``
  3. create a dedicated environment with ``conda env create -f protopipe_environment.yml``
  4. activate it with ``conda activate protopipe``
  5. install *protopipe* itself with ``python setup.py install``.

Instructions for advanced users
-------------------------------

If you want to use *protopipe* and also contribute to its development, follow these steps:

  1. Fork the official `repository <https://github.com/cta-observatory/protopipe>`_ has explained `here <https://help.github.com/en/articles/fork-a-repo>`__ (follow all the instructions)
  2. now your local copy is linked to your remote repository (**origin**) and the official one (**upstream**)
  3. execute points 3 and 4 in the instructions for basic users
  4. install *protopipe* itself in developer mode with ``python setup.py develop``

When you want to fix a bug or develop something new:

  1. update your **local** *master* branch with `git pull upstream master`
  2. create and move to a new **local** branch from your **local** *master* with `git checkout -b your_branch`
  3. develop inside it
  4. push it to *origin*, thereby creating a copy of your branch also there
  5. continue to develop and push until you feel ready
  6. start a *pull request* using the web interface from *origin/your_branch* to *upstream/master*

    1. wait for an outcome
    2. if necessary, you can update or fix things in your branch because now everything is traced (**local/your_branch** --> **origin/your_branch** --> **pull request**)

.. Note::

  If your developments take a relatively long time, consider to update periodically your **local** *master* branch.

  If in doing this you see that the files on which you are working on have been modified *upstream*,

    * move into your **local** branch,
    * merge the new master into your branch ``git merge master``,
    * resolve eventual conflicts
    * push to origin

  In this way, your pull request will be up-to-date with the master branch into which you want to merge your changes.
  If your changes are relatively small and `you know what you are doing <https://www.atlassian.com/git/tutorials/merging-vs-rebasing>`_, you can use ``git rebase master``, instead of merging.

How to?
=======
For this pipeline prototype, in order to build an analysis to estimate
the performance of the instruments, a user will follows the following steps:

 1. Energy estimator

  * produce a table with gamma-ray image information with pipeline utilities (:ref:`pipeline`)
  * build a model with mva utilities (:ref:`mva`)

 2. Gamma hadron classifier

  * produce tables of gamma-rays and hadrons with image informations with pipeline utilities (:ref:`pipeline`)
  * build a model with mva utilities (:ref:`mva`)

 3. DL2 production

  * produce tables of gamma-rays, hadrons and electrons with event informations with pipeline utilities (:ref:`pipeline`)

 4. Estimate performance of the instrument

  * find the best cutoff in gammaness/score, to discriminate between signal
    and background, as well as the angular cut to obtain the best sensitivity
    for a given amount of observation time and a given template for the
    source of interest (:ref:`perf`)
  * compute the instrument response functions, effective area,
    point spread function and energy resolution (:ref:`perf`)
  * estimate the sensitivity (:ref:`perf`)


Documentation
=============

.. toctree::
   :maxdepth: 1

   pipeline/index
   mva/index
   perf/index
   scripts/index

.. _ctapipe installation: https://cta-observatory.github.io/ctapipe/getting_started/index.html#step-4-set-up-your-package-environment
.. _ctapipe: https://cta-observatory.github.io/ctapipe/
.. _gammapy: https://gammapy.org/
.. _pywi: http://www.pywi.org/
.. _pywi-cta: http://cta.pywi.org/
