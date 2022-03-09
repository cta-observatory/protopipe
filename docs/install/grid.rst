.. _install-grid:

===========================
Interface to the DIRAC grid
===========================

Requirements
============

.. _base_env_protopipe_CTADIRAC:

Base environment
----------------

Aside from *protopipe* itself,
the software/packages required to work on the DIRAC grid are the following,

- `DIRAC <https://dirac.readthedocs.io/en/latest/>`_
- `CTADIRAC <https://gitlab.cta-observatory.org/cta-computing/dpps/CTADIRAC>`_
- `VOMS <https://italiangrid.github.io/voms/>`_
- `pytables <https://www.pytables.org/>`_
- `pyyaml <https://pyyaml.org/>`_

Below you can find a recipe for the **base** environment into which install *protopipe*
and its interface.  
Recipes are also stored in the `root directory of the interface <https://github.com/HealthyPear/protopipe-grid-interface>`_.

.. code-block:: yaml

  name: protopipe-CTADIRAC
  channels:
    - conda-forge
  dependencies:
    - python>=3.8
    - pip
    # required by the interface
    - dirac-grid
    - diracgrid::fts3
    - voms
    - pytables
    - pyyaml
    # required for ctapipe 0.11.0 based installation
    # will work anyway if such dependency is upgraded
    - gammapy=0.18
    - eventio
    - pip:
        - CTADIRAC

After its creation you have to install first *protopipe* (:ref:`install_protopipe`)
and then its interface (:ref:`install-interface`).

An overview of the versioning between the released sofware is described below (:ref:`versioning`)

DIRAC GRID certificate
----------------------

In order to access the GRID utilities you will need a certificate associated with an
account.

You can find all necessary information at
`this <https://forge.in2p3.fr/projects/cta_dirac/wiki/CTA-DIRAC_Users_Guide#Prerequisites>`_
Redmine wikipage.

.. _install-interface:

Installation
============

Also the interface to the DIRAC grid can be installed both in development mode
and as a released package.

Released version
----------------

.. important::

  After the Python3 upgrade of DIRAC and CTADIRAC,
  the interface installation and usage have changed considerably,
  while its relation with *protopipe* has only improved.
  It is **extremely** unlikely that you will ever need to work with a version older than v0.4.0,
  but if this were to happen, please check older versions of this documentation
  either from readthedocs or from the repository of *protopipe*.

You can install install it as a Python3-based package in your environment like so,

``pip install git+https://github.com/HealthyPear/protopipe-grid-interface@vx.y.z``

The following table refers to all versions and their compatibility with *protopipe*.

.. list-table:: Versioning
    :name: versioning
    :widths: 25 25
    :header-rows: 0

    * - protopipe
      - GRID interface
    * - v0.5.X
      - v0.4.X
    * - v0.4.X
      - v0.3.X
    * - v0.3.X
      - v0.2.X
    * - v0.2.X
      - v0.2.X

The latest released version of the interface is also compatible with
the development version of *protopipe*.

.. _install-grid-dev:

Development version
-------------------

This version is:
- always compatible with the development version of *protopipe*,
- possibly compatible with the latest release of *protopipe*,

The procedure to install with this version is similar to the same one
for *protopipe*:

- ``git clone https://github.com/HealthyPear/protopipe-grid-interface.git``
- ``cd protopipe-grid-interface``
- ``pip install -e '.[all]'``

Setup the working environment
=============================

In order to be able to download and upload files from and to the DIRAC grid
you need to initialize the Virtual Organisation Membership Service (VOMS).

This is a one time operation to be perfomed after the environment creation and activation:

.. code-block:: shell

   conda env config vars set X509_CERT_DIR=$CONDA_PREFIX/etc/grid-security/certificates
   conda env config vars set X509_VOMS_DIR=$CONDA_PREFIX/etc/grid-security/vomsdir
   conda env config vars set X509_VOMSES=$CONDA_PREFIX/etc/grid-security/vomses
   conda activate protopipe-CTADIRAC

Also only the first time, in order to use the CTADIRAC production instance,
you should configure your client using the ``dirac-configure`` command.
You will be asked to generate your proxy and then to choose the ``Setup`` and the ``Configuration`` server.
You need to choose the default values.

.. warning::
  The configuration system defaults could lack redundance depending on the version of CTADIRAC.
  It is suggested to check or edit the configuration file that you can find inside your conda enviroment
  under ``etc/dirac.cfg`` like the following,

  .. code-block::

    DIRAC
    {
    Setup = CTA
    Configuration
    {
      Servers = dips://dcta-servers02.pic.es:9135/Configuration/Server
      Servers += dips://dcta-servers02.pic.es:9135/Configuration/Server
      Servers += dips://dcta-agents02.pic.es:9135/Configuration/Server
      Servers += dips://ccdcta-server04.in2p3.fr:9135/Configuration/Server
      Servers += dips://ccdcta-server05.in2p3.fr:9135/Configuration/Server
      Servers += dips://ccdcta-web01.in2p3.fr:9135/Configuration/Server
    }
    Security
    {
      UseServerCertificate = no
    }
    }
    LocalInstallation
    {
    Setup = CTA
    ConfigurationServer = dips://dcta-servers02.pic.es:9135/Configuration/Server
    ConfigurationServer += dips://dcta-agents02.pic.es:9135/Configuration/Server
    ConfigurationServer += dips://ccdcta-server04.in2p3.fr:9135/Configuration/Server
    ConfigurationServer += dips://ccdcta-server05.in2p3.fr:9135/Configuration/Server
    ConfigurationServer += dips://ccdcta-web01.in2p3.fr:9135/Configuration/Server
    SkipCAChecks = True
    }

For the subsequent times, it will be sufficient to generate the proxy 
with ``dirac-proxy-init`` (it will lasts up to 24h if you don't exit the environment before).

Now you can proceed with the analysis workflow (:ref:`use-grid`).
