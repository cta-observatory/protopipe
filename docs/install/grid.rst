.. _install-grid:

Grid environment
================

General requirements
--------------------

Regardless of what operating system you are using, in order to be able to use
*protopipe* on the grid you will need to be able to interface with it.

In order to be able to interface with the DIRAC grid, you should have already
followed the initial steps described
`here <https://forge.in2p3.fr/projects/cta_dirac/wiki/CTA-DIRAC_Users_Guide>`__.

In the following it is then assumed that you have:

* a Grid identity and a CTA VO registration,
* a certificate and a user key in the hidden folder ``$HOME/.globus``.

Installation instructions
-------------------------

The installation instructions are the following:

* only macos/Windows should do this first

  * initialize a virtual machine with Vagrant using *this VagrantFile*,
  * change the first arguments of lines 47 to 50 to suit your local setup,
  * ``vagrant up``
  * ``vagrant ssh``

* then all users

  * ``singularity pull *containerURL_TO BE ADDED*``

.. Note::
  For Linux users, the host's $HOME coincides with the guests' one by default; you
  will need to link the source codes for protopipe and the grid there.
  For macos/Windows users, this has been set when you edited the VagrantFile.

From this point every user should be able to enter the singularity container
with ``singularity shell ctadirac_protopipe.simg``.
The container provides the latest version of CTADIRAC and the necessary python
modules for the protopipe-grid code to run.

Now you should be able to setup your analysis on the grid (:ref:'use-grid').
Note that two different environments must be use to deal with ctapipe
(Python >=3.5, conda environment)
and the grid utilities (Python 2.7, built-in environment).

.. Note::

  The paths already contained in the VagrantFile correspond to the local setup
  used by the author, but it can be changed to suit best your situation.

  In particular the `data` folder where the VagrantFile is stored is a shared
  folder between host and guest, and it can be used to store both the results
  of the simulations and the lists of MonteCarlo files to be used.
