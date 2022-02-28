.. _install-grid:

===========================
Interface to the DIRAC grid
===========================

.. contents::
   :local:

Requirements
============

.. _base_env_protopipe_CTADIRAC:

Base environment
----------------

The software/packages required to work with the DIRAC interface for protopipe
are the following,

- `DIRAC <https://dirac.readthedocs.io/en/latest/>`_
- `CTADIRAC <https://gitlab.cta-observatory.org/cta-computing/dpps/CTADIRAC>`_
- `VOMS <https://italiangrid.github.io/voms/>`_
- `pytables <https://www.pytables.org/>`_
- `pyyaml <https://pyyaml.org/>`_

.. note:: Python version
  Even if *protopipe* itself allows for Python 3.7, the minimum version required to use
  Dirac is 3.8.

For convenience, a ready-to-use conda environment recipe is reported below.
You can also find it in the `root directory of the interface <https://github.com/HealthyPear/protopipe-grid-interface>`_.
You can create the corresponding environment with a command like this,

``conda env create -f my_env_recipe.yml -n my_env``

.. code-block:: yaml

  name: protopipe-CTADIRAC
  channels:
    - conda-forge
  dependencies:
    - python>=3.8
    - pip
    - dirac-grid
    - voms
    - pytables
    - pyyaml
    - eventio # required for protopipe with ctapipe <0.12.0
    - gammapy=0.18 # required for protopipe with ctapipe <0.12.0
    - pip:
        - CTADIRAC

In this environment you can then install first protopipe and then the interface.

If you want to install them in their released versions, integrate the ``pip`` section
with,

- ``protopipe``/``protopipe=x.y.z`` for a latest/specific released version of protopipe,
- ``git+https://github.com/HealthyPear/protopipe-grid-interface@vx.y.z`` for a released version of its interface.

An overview of the versioning between the two sofware is described below (:ref:`versioning`)

If you are a developer, or you want to work with the development version,
please refer to the development installation instructions for protopipe (:ref:`install-development`)
and its interface (:ref:`install-grid-dev`).

DIRAC GRID certificate
----------------------

In order to access the GRID utilities you will need a certificate associated with an
account.

You can find all necessary information at
`this <https://forge.in2p3.fr/projects/cta_dirac/wiki/CTA-DIRAC_Users_Guide#Prerequisites>`_
Redmine wikipage.

The interface
=============

Getting a released version
--------------------------

For versions >=0.4.0 you can install it as a Python3-based package in your environment as explained
above.

The following table refers to all versions and their compatibility with _protopipe_.

.. list-table:: compatibility between *protopipe* and its interface
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

.. warning::

  After the Python3 upgrade of DIRAC and CTADIRAC,
  the interface installation and usage have changed considerably,
  while its relation with _protopipe_ has only improved.
  It is very unlikely that you will ever need to work with a version older than v0.4.0,
  but if this were to happen, please check older versions of this documentation
  either from readthedocs or from the repository of _protopipe_.

.. _install-grid-dev:

Getting the development version
-------------------------------

This version is:
- always compatible with the development version of *protopipe*,
- possibly compatible with the latest release of *protopipe*,

The procedure to install with this version is similar to the same one
for _protopipe_:

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
  The defaults right now are lacking redundance in the configuration system.
  It is suggested to edit the configuration file that you can find inside your conda enviroment
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
