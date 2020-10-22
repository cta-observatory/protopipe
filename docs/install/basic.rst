.. _install-basic:

Basic users
===========

If you are a user with no interest in developing *protopipe*, you can start by
downoading the `latest released version <https://github.com/cta-observatory/protopipe/releases>`__

Steps for installation:

  1. uncompress the file which is always called *protopipe-X.Y.Z* depending on version,
  2. enter the folder ``cd protopipe-X.Y.Z``
  3. create a dedicated environment with ``conda env create -f environment.yml``
  4. activate it with ``conda activate protopipe``
  5. install *protopipe* itself with ``python setup.py install``.

.. warning::
  Given that *protopipe* is undergoing fast development, it is likely that you
  will benefit more from a more recent version of the code for now.

  The development version could disrupt functionalities that were working for
  you, but the latest released version could lack some of those you need.

  To install the latest development version go to :ref:`install-developer`.

Next steps:

 * get accustomed to the basic pipeline workflow (:ref:`use-pipeline`),
 * then make your own complete analysis (:ref:`use-grid`),
 * for bugs and new features, please contribute to the project (:ref:`contribute`).
