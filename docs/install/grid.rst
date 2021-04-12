.. _install-grid:

Grid environment
================

Requirements
------------

* credentials and certificate for using the GRID (follow `these instructions <https://forge.in2p3.fr/projects/cta_dirac/wiki/CTA-DIRAC_Users_Guide>`__.)
* `Vagrant <https://www.vagrantup.com/>`_

Getting a released version
--------------------------

You can find the latest released version `here <https://github.com/cta-observatory/protopipe/releases>`__

.. list-table:: compatibility between *protopipe* and its interface
    :widths: 25 25
    :header-rows: 0

    * - protopipe
      - GRID interface
    * - v0.2.X
      - v0.2.X
    * - v0.3.X
      - v0.2.X
    * - v0.4.X
      - v0.3.X

The latest released version of the GRID interface is also compatible with
the development version of *protopipe*.

Getting the development version
-------------------------------

This version is always compatible *only* with the development version of *protopipe*.

``git clone https://github.com/HealthyPear/protopipe-grid-interface.git``

Setup the working environment
-----------------------------

1. create and enter a folder where to work,
2. copy the ``VagrantFile`` from the interface
3. edit lines from 48 to 59 according to your local setup
4. ``vagrant up && vagrant ssh``
5. ``singularity pull --name CTADIRAC_with_protopipe.sif shub://HealthyPear/CTADIRAC``
6. ``singularity shell CTADIRAC_with_protopipe.sif``

From here,

- activate the GRID environment with ``dirac-proxy-init``
- the ``shared_folder`` contains the folders

  - ``analyses`` to store all your analyses
  - ``productions`` to store lists of simulated files

Now you can proceed with the analysis workflow (:ref:`use-grid`).
