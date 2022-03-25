.. protopipe documentation master file, created by
   sphinx-quickstart on Sat Feb 29 01:44:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to protopipe's documentation!
=====================================

`Protopipe` is a pipeline prototype for the `Cherenkov Telescope Array <https://www.cta-observatory.org/>`__
(CTA).  
It is based on the `ctapipe <https://cta-observatory.github.io/ctapipe>`__ and `pyirf <https://github.com/cta-observatory/pyirf>`__
libraries among other third-party Python packages.  
The package has been developed and tested in the department of astrophysics at
`CEA-Saclay/IRFU <http://irfu.cea.fr/dap/en/index.php>`__,
but since September 2019 has been open for development to all members of the CTA consortium.

The source code is hosted on a `GitHub repository <https://github.com/cta-observatory/protopipe>`__ 
to which this documentation is linked.  
It is planned to migrate it to the CTAO-GitLab installation under the ASWG group
repository (`link <https://gitlab.cta-observatory.org/cta-consortium/aswg>`__).

Current performance is stored internally at `GitLab <http://cccta-dataserver.in2p3.fr/data/protopipe/results/html/>`__.

.. warning::
  This is not yet stable code, so expect large and rapid changes.

Resources
---------

- Source code:

  - `protopipe <https://github.com/cta-observatory/protopipe>`__
  - `DIRAC grid interface <https://github.com/HealthyPear/protopipe-grid-interface>`__

- Documentation:

  - `GitHub Pages <https://cta-observatory.github.io/protopipe>`__ (only development version)
  - `readthedocs <https://protopipe.readthedocs.io/en/latest/>`__ (also latest releases)

- Performance results: `GitLab <http://cccta-dataserver.in2p3.fr/data/protopipe/results/html/>`__

- Slack channels:

  - `#protopipe <https://cta-aswg.slack.com/archives/CPTN4U7U7>`__
  - `#protopipe_github <https://cta-aswg.slack.com/archives/CPUSPPHST>`__
  - `#protopipe-grid <https://cta-aswg.slack.com/archives/C01FWH8E0TT>`__

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
- v0.5.0 : TBD
- v0.4.0 : |doi_v0.4.0|
- v0.3.0 : |doi_v0.3.0|

.. toctree::
    :hidden:
    
    install/index
    usage/index
    reference/index
    contribute/index
    AUTHORS
    CHANGELOG

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`