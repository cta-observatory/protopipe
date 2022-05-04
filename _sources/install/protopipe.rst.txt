.. _install_protopipe:

*********
protopipe
*********

.. important::
  If you already know you want to use *protopipe* on the *DIRAC* grid, please start
  from the installation based on its interface (:ref:`protopipe-CTADIRAC base environment <base_env_protopipe_CTADIRAC>`)
  and then come back here.

There are two different ways to install ``protopipe``,

* if you just want to use it as it is (:ref:`install-release`),
* or if you also want to develop it (:ref:`install-development`).

To perform full analyses, you will need some computational power in order to 
produce enough data files for model and performance estimation.
This can be accomplished through the use of the DIRAC computing grid (:ref:`install-grid`).

.. _install-release:

Released version
================

To install the latest released version it is sufficient to install the
package from ``PyPI`` with ``pip install protopipe``.

If you prefer to work from an Miniconda/Anaconda virtual environment you can create it by downloading
`the recipe <https://github.com/cta-observatory/protopipe/blob/master/environment_latest_release.yml>`__
and issue the following command from the base environment,

``conda env create -f environment_latest_release.yml``

.. _install-development:

Development version
===================

1. `fork <https://help.github.com/en/articles/fork-a-repo>`__ the `repository <https://github.com/cta-observatory/protopipe>`_
2. it is recommended to create a virtual environment (Miniconda/Anaconda users can use the ``environment_development.yml`` stored at the root of the cloned repository)
3. ``pip install -e '.[all]'``

The ``all`` keyword will install all extra requirements,
which can be also installed separately using ``tests`` and/or ``docs``.

Next steps
==========

* get accustomed to the basic pipeline workflow (:ref:`use-pipeline`),
* make your own complete analysis (:ref:`use-grid`),
* learn how to contribute to the project (:ref:`contribute`).

