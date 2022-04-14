.. protopipe documentation master file, created by
   sphinx-quickstart on Sat Feb 29 01:44:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================================
Welcome to protopipe's documentation!
=====================================

.. |CI| image:: https://github.com/cta-observatory/protopipe/workflows/CI/badge.svg?branch=master
  :target: https://github.com/cta-observatory/protopipe/actions?query=workflow%3ACI
.. |codacy|  image:: https://app.codacy.com/project/badge/Grade/cb95f2eee92946f2a68acc7b103f843c
  :target: https://www.codacy.com/gh/cta-observatory/protopipe?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cta-observatory/protopipe&amp;utm_campaign=Badge_Grade
.. |coverage| image:: https://codecov.io/gh/cta-observatory/protopipe/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/cta-observatory/protopipe
.. |documentation| image:: https://readthedocs.org/projects/protopipe/badge/?version=latest
  :target: https://protopipe.readthedocs.io/en/latest/?badge=latest
.. |doilatest| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4586754.svg
  :target: https://doi.org/10.5281/zenodo.4586754
.. |pypi| image:: https://badge.fury.io/py/protopipe.svg
    :target: https://badge.fury.io/py/protopipe


|CI| |codacy| |coverage| |documentation| |doilatest| |pypi|

`Protopipe` is a pipeline prototype for the `Cherenkov Telescope Array <https://www.cta-observatory.org/>`__
(CTA) based on the `ctapipe <https://cta-observatory.github.io/ctapipe>`__ and `pyirf <https://github.com/cta-observatory/pyirf>`__
libraries among other third-party Python packages.

.. warning::
  This is not yet stable code, so expect large and rapid changes.

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6

    ---
    :img-top: _static/news.svg

    What's new
    ^^^^^^^^^^

    Checkout the latest features and bug fixes from the last release
    and the development version.

    +++

    .. link-button:: CHANGELOG
            :type: ref
            :text: To the changelog
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/chart-column-solid.svg

    Performance
    ^^^^^^^^^^^

    CTA performance results produced with *protopipe*  and deployed
    from CTAO GitLab to a website on a CC-IN2P3 server.

    +++

    .. link-button:: http://cccta-dataserver.in2p3.fr/data/protopipe/results/html/
            :type: url
            :text: To the performance results
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/install.svg

    Getting started
    ^^^^^^^^^^^^^^^

    Find the solution that best suits your use-case and proceed
    to installation and environment setup.

    +++

    .. link-button:: install
            :type: ref
            :text: To the installation guides
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/book-open-solid.svg

    User guide
    ^^^^^^^^^^

    Understand the key concepts and go through your first complete analysis
    using the DIRAC grid.

    +++

    .. link-button:: usage
            :type: ref
            :text: To the user guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/api.svg

    API reference
    ^^^^^^^^^^^^^

    Detailed description of the functions and classes
    defined in *protopipe*. It is assumed you understand the key concepts.

    +++

    .. link-button:: reference
            :type: ref
            :text: To the reference guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/code-solid.svg

    Developer guide
    ^^^^^^^^^^^^^^^

    Found a bug? A missing functionality?
    Join the development effort and help improve *protopipe*.

    +++

    .. link-button:: contribute
            :type: ref
            :text: To the development guide
            :classes: btn-block btn-secondary stretched-link
    
    ---
    :img-top: _static/slack.svg

    Slack channels
    ^^^^^^^^^^^^^^

    Join the Slack channels to be up-to-date on development
    and grid services.

    `#protopipe <https://cta-aswg.slack.com/archives/CPTN4U7U7>`__  

    `#protopipe_github <https://cta-aswg.slack.com/archives/CPUSPPHST>`__  

    `#protopipe-grid <https://cta-aswg.slack.com/archives/C01FWH8E0TT>`__

    ---
    :img-top: _static/citing.svg

    Citing
    ^^^^^^

    If you make use of *protopipe* for a publication,
    please consider to cite it.

    +++

    .. link-button:: cite
            :type: ref
            :text: To the citation instructions
            :classes: btn-block btn-secondary stretched-link

.. toctree::
    :hidden:
    
    install/index
    usage/index
    reference/index
    contribute/index
    AUTHORS
    CHANGELOG