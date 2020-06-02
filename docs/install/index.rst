.. _install:

Installation
============

The only requirement is an Anaconda (or Miniconda) installation which supports
Python 3.

There are two different ways to install protopipe,

* if you just want to use it as it is (:ref:`install-basic`),
* or if you also want to develop it (:ref:`install-developer`).

.. Note::

  For a faster use, edit your preferred login script (e.g. ``.bashrc`` on Linux or
  ``.profile`` on macos) with a function that initializes the environment.
  The following is a minimal example using Bash.

  .. code-block:: bash

    alias protopipe="protopipe_init"

    function protopipe_init() {

        conda activate protopipe # Then activate the protopipe environment
        export PROTOPIPE=$WHEREISPROTOPIPE/protopipe # A shortcut to the scripts folder

    }

.. toctree::
    :hidden:
    :maxdepth: 1

    basic
    developers
