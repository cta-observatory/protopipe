.. _install-release:

Released version
================

To install the latest released version it is sufficient to install the
package from ``PyPI`` with ``pip install protopipe``.

If you prefer to work from an Anaconda virtual environment you can create it with,

``conda env create -f environment_latest_release.yml``

For previous releases,

1. download the corresponding tarball stored `here <https://github.com/cta-observatory/protopipe/releases>`__
2. ``conda env create -f environment.yml -n protopipe-X.Y.Z``
3. ``conda activate protopipe-X.Y.Z``
4. ``pip install .``

Next steps:

  * get accustomed to the basic pipeline workflow (:ref:`use-pipeline`),
  * make your own complete analysis (:ref:`use-grid`),
  * for bugs and new features, please contribute to the project (:ref:`contribute`).
