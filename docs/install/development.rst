.. _install-development:

Development version
===================

  1. `fork <https://help.github.com/en/articles/fork-a-repo>`__ the `repository <https://github.com/cta-observatory/protopipe>`_
  2. create a virtual environment (Anaconda users can use the ``environment_development.yml`` file)
  3. ``pip install -e '.[all]'``
  
  The ``all`` keyword will install all extra requirements,
  which can be also installed separately using ``tests`` and/or ``docs``.

Next steps:

  * get accustomed to the basic pipeline workflow (:ref:`use-pipeline`),
  * make your own complete analysis (:ref:`use-grid`),
  * learn how to contribute to the project (:ref:`contribute`).
