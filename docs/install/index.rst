.. _install:

Installation
============

Requirements:

- ``Python >= 3.7``
- ``pip``

It is recommended to work within a virtual environment, using for example
``venv`` or Anaconda's ``conda``. 

.. note::
  In case you want to use Anaconda it is recommended to use
  `mamba <https://github.com/mamba-org/mamba#readme>`__
  for improved speed during the creation of the virtual environment.

There are two different ways to install ``protopipe``,

* if you just want to use it as it is (:ref:`install-release`),
* or if you also want to develop it (:ref:`install-development`).

In both cases, if you want to perform a full analysis, you will need some
computational power in order to produce enough
data files for model and performance estimation.
This can be accomplished through the use of a GRID environment.

After installing ``protopipe``,

* install the code necessary to interface it with the grid (:ref:`install-grid`),
* use protopipe on the grid (:ref:`use-grid`).

.. toctree::
    :hidden:
    :maxdepth: 1

    release
    development
    grid
