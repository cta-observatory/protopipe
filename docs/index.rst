.. protopipe documentation master file, created by
   sphinx-quickstart on Sat Feb 29 01:44:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to protopipe's documentation!
=====================================

`Protopipe` is a pipeline prototype for the `Cherenkov Telescope Array <https://www.cta-observatory.org/>`__
(CTA) based on the `ctapipe <https://cta-observatory.github.io/ctapipe>`__ library.
The package has been developed and tested in the department of astrophysics at
`CEA-Saclay/IRFU <http://irfu.cea.fr/dap/en/index.php>`__,
but since Sep 23, 2019 is also open for development by other members of the CTA consortium.

The source code is currently hosted on a `GitHub repository <https://github.com/cta-observatory/protopipe>`__ 
to which this documentation is linked.  
It will soon migrate to the CTAO-GitLab installation under the ASWG group
repository (`link <https://gitlab.cta-observatory.org/cta-consortium/aswg>`__).

Current performance is stored internally at `this RedMine page <https://forge.in2p3.fr/projects/benchmarks-reference-analysis/wiki/Protopipe_performance_data>`__.

.. warning::
  This is not yet stable code, so expect large and rapid changes.

Citing this software
--------------------

.. |doilatest| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4586754.svg
  :target: https://doi.org/10.5281/zenodo.4586754
.. |doi_v0.4.0| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4586755.svg
  :target: https://doi.org/10.5281/zenodo.4586755
.. |doi_v0.3.0| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4303996.svg
  :target: https://doi.org/10.5281/zenodo.4303996

If you use a released version of this software for a publication,
please cite it by using the corresponding DOI.

Please, check the development version of the README for up-to-date links.

- latest : |doilatest|
- v0.4.0 : |doi_v0.4.0|
- v0.3.0 : |doi_v0.3.0|


.. _protopipe_intro:
.. toctree::
   :caption: Overview
   :maxdepth: 1

   install/index
   usage/index
   contribute/index
   AUTHORS
   CHANGELOG

.. _protopipe_structure:
.. toctree::
   :caption: Structure
   :maxdepth: 1

   pipeline/index
   mva/index
   perf/index
   scripts/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`