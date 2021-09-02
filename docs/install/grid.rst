.. _install-grid:

Grid environment
================

Requirements
------------

DIRAC GRID certificate
++++++++++++++++++++++

In order to access the GRID utilities you will need a certificate associated with an
account.

You can find all necessary information 
`here <https://forge.in2p3.fr/projects/cta_dirac/wiki/CTA-DIRAC_Users_Guide#Prerequisites>`_.

Source code for the interface
+++++++++++++++++++++++++++++

.. warning::
  Usage of the pipeline on an infrastucture different than the DIRAC grid has not been fully tested.
  This interface code is higly bound to DIRAC, but the scripts which manage download, merge and upload of files
  could be easily adapted to different infrastructures.

Getting a released version
^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This version is always compatible *only* with the development version of *protopipe*.

``git clone https://github.com/HealthyPear/protopipe-grid-interface.git``

Container and options for containerization
------------------------------------------

.. note::
  Any of the following containerization choices constitutes a requirement.

- **Single user working from personal machine**

  The *Docker* container should be enough.

- **User working on a more or less shared environment (HPC machine or server)**

  In case you are not allowed to use *Docker* for security reasons, another supported option is *Singularity*.

  - on *Linux* make sure *Singularity* is installed and accessible to your user,
  - on *Windows* or *macos*, you will need to install *Vagrant*.

Docker
++++++

The container used by the interface requires the 
`installation of Docker <https://docs.docker.com/get-docker/>`_.

To enter the container (and the first time downloading the image),

``docker run --rm -v $HOME/.globus:/home/dirac/.globus -v $PWD/shared_folder:/home/dirac/shared_folder -v [...]/protopipe:/home/dirac/protopipe -v [...]/protopipe-grid-interface:/home/dirac/protopipe-grid-interface -it ctadirac/client``

where ``[...]`` is the path of your source code on the host and the ``--rm`` 
flag will erase the container at exit
to save disk space (the data stored in the ``shared_folder`` won't disappear).
Please, refer to the Docker documentation for other use cases.

.. note::
  In case you are using a released version of *protopipe*, there is no container
  at the moment and the GRID environment based on CTADIRAC still requires Python2.
  In this case you can link the source code folder from your python environment
  installation on the host just like you would do with the development
  version (``import protopipe; protopipe.__path__``).

.. warning::
  If you are using *macos* you could encounter some disk space issues.
  Please check `here <https://docs.docker.com/docker-for-mac/space/>`_ and
  `here <https://djs55.github.io/jekyll/update/2017/11/27/docker-for-mac-disk-space.html>`_
  on how to manage disk space.

Vagrant
+++++++

.. note::
  Only required for users that want to use a *Singularity*
  container on a *macos* and *Microsoft Windows* machine.

All users, regardless of their operative systems, can use this interface via
`Vagrant <https://www.vagrantup.com/>`_. 

The *VagrantFile* provided with the interface code allows to download a virtual 
machine in form of a *Vagrant box* which will host the actual container.

The user needs to,

1. copy the ``VagrantFile`` from the interface
2. edit lines from 48 to 59 according to the local setup
3. enter the virtual machine with``vagrant up && vagrant ssh``

The *VagrantFile* defines creates automatically also the ``shared_folder``
used by the interface to setup the analysis.

Singularity
+++++++++++

.. warning::
  Support for *Singularity* has been dropped by the mantainers of *CTADIRAC*.
  The following solutions have not been tested in all possible cases.

- **macos / Microsoft Windows**

  `Singularity <https://sylabs.io/docs/>`_ is already installed and ready to use from the *Vagrant box* 
  obtained by using the *VagrantFile*.

- **Linux**
  
  users that do not want to use *Vagrant* will need to have *Singularity* installed
  on their systems and they will need to edit their own environment accordingly.

  For pure-*Singularity* users (aka on Linux machines without *Vagrant*) 
  bind mounts for *protopipe*, its grid interface and the shared_folder 
  will work in the same way: ``--bind path_on_host:path_on_container``.

The DIRAC grid certificate should be already available since *Singularity* 
mounts the user's home by default.
For more details, please check e.g. 
`system-defined bind paths <https://sylabs.io/guides/3.8/user-guide/bind_paths_and_mounts.html#system-defined-bind-paths>`_.

Depending on the privileges granted on the host there are 2 ways to get a working container.

Using the CTADIRAC Docker image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Method #1**

Provided you have at least *Singularity 3.3*, you can pull directly the CTADIRAC Docker image from *DockerHub*,
but you will need to use the ``fakeroot`` mode.
This mode grants you root privileges only *inside* the container.

``singularity build --fakeroot ctadirac_client_latest.sif docker://ctadirac/client``

``singularity shell --fakeroot ctadirac_client_latest``

``. /home/dirac/dirac_env.sh``

**Method #2**

You shouldn't need root privileges for this to work (not throughly tested, though),

``singularity build --sandbox --fix-perms ctadirac_client_latest.sif docker://ctadirac/client``

``singularity shell ctadirac_client_latest``

``. /home/dirac/dirac_env.sh``

Building the Singularity image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Support for *Singularity* has been dropped by the mantainers of *CTADIRAC*,
but the recipe for the container has been saved here.

In this case you won't need to do ``. /home/dirac/dirac_env.sh``: the 
commands will be already stored in your ``$PATH``.

.. warning::
  The recipe ``CTADIRAC_singularity`` is maintained by the author; if any bug arises,
  reverting to the methods described above (if possible) will provide you with a working environment.

If you have root privileges you can just build your own image with,

``singularity build ctadirac_client_latest.sif CTADIRAC_singularity``

otherwise you have to either,

- revert to the ``--fakeroot`` mode 
  (use it also to enter the container just like the methods above)

- build the image remotely at ``https://cloud.sylabs.io`` using the ``--remote`` flag
  (for this you will need to interface with that servce to generate an access token)

Setup the working environment
-----------------------------

The CTADIRAC container doesn't provide everything *protopipe* needs,
but this can be solved easily by issuing the following command inside the container's home directory,

``source protopipe-grid-interface/setup.sh``

This will not only install the missing Python packages,
but also provide convenient environment variables ``$INTERFACE`` and ``$PROTOPIPE``
for the source code.

From here,

- activate the GRID environment with ``dirac-proxy-init``
- the ``shared_folder`` contains the folders

  - ``analyses`` to store all your analyses
  - ``productions`` to store lists of simulated files

Now you can proceed with the analysis workflow (:ref:`use-grid`).
