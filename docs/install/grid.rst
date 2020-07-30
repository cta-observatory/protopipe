.. _install-grid:

Grid environment
================

General requirements
--------------------

Regardless of what operating system you are using, in order to be able to use
*protopipe* on the grid you will need to be able to interface with it.

Go through the initial steps described
`here <https://forge.in2p3.fr/projects/cta_dirac/wiki/CTA-DIRAC_Users_Guide>`__.

In the following it is then assumed that you have:

* a Grid identity and a CTA VO registration,
* a certificate and a user key in the hidden folder ``$HOME/.globus``

Installation instructions
-------------------------

The installation instructions are the following:

* get the interface code (``git clone https://github.com/HealthyPear/protopipe-grid-interface.git``),
* create and enter a folder where to work,
* **(only Windows/macos users)** copy the ``VagrantFile`` from the parent folder of `protopipe-grid-interface`,
* **(only Windows/macos users)** edit the first arguments of lines from 47 to 50 according to your local setup,
* create the `data` shared folder to interact externally with the analysis products,
* **(only Windows/macos users)** enter in the virtual environment (``vagrant up && vagrant ssh``),
* get the Singularity container (``singularity pull --name CTADIRAC_with_protopipe.sif shub://HealthyPear/CTADIRAC``),
* enter in it (``singularity shell CTADIRAC_with_protopipe.sif``).

From here you should be able to activate the DIRAC environment (``dirac-proxy-init``).
The container provides the latest version of CTADIRAC and the necessary python
modules for the `protopipe-grid` code to run.
Now you can proceed with the analysis workflow (:ref:`use-grid`).

.. Note::
  For Linux users, the host's $HOME coincides with the guests' one by default.
  For macos/Windows users, this has been set when you edited the VagrantFile.
