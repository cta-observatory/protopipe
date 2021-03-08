.. _install-development:

Development version
===================

  1. `fork <https://help.github.com/en/articles/fork-a-repo>`__ the `repository <https://github.com/cta-observatory/protopipe>`_
  2. create and enter a basic virtual environment (or use the ``environment.yaml`` file)
  3. ``pip install -e '.[all]'``
  
  The ``all`` keyword will install all extra requirements,
  which can be also installed separately using ``tests`` and ``docs``.

Next steps:

  * get accustomed to the basic pipeline workflow (:ref:`use-pipeline`),
  * make your own complete analysis (:ref:`use-grid`),
  * learn how to contribute to the project (:ref:`contribute`).
