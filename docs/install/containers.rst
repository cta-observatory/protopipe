.. _containers:

Options for containerization
----------------------------

The python-based installation should suffice for the great majority of users,
but if necessary some options for containerization are provided.

All the required material is stored in the root folder
of the grid interface repository under `containers`.

Some use-cases could be the following,

- **Single user working from a personal machine**

  The *Docker* container should be enough.

- **User working on a more or less shared environment (HPC machine or server)**

  In case you are not allowed to use *Docker* for security reasons, another option is *Singularity*.

  - on *Linux* make sure *Singularity* is installed and accessible to your user,
  - on *Windows* or *macos*, you will need to install *Vagrant*.

.. warning::
  After the upgrade to python3 of DIRAC and CTADIRAC, focus was given only to the Docker solution.
  Singularity and Vagrant-based solution are not currenlty supported.


Docker
++++++

The container used by the interface requires the `installation of Docker <https://docs.docker.com/get-docker/>`_.

To enter the container (and the first time downloading the image),

``docker run --rm -v $HOME/.globus:/home/dirac/.globus -v $PWD/shared_folder:/home/dirac/shared_folder -v [...]/protopipe:/home/dirac/protopipe -v [...]/protopipe-grid-interface:/home/dirac/protopipe-grid-interface -it ctadirac/client``

where ``[...]`` is the path of your source code on the host and the ``--rm`` flag will erase the container at exit
to save disk space (the data stored in the ``shared_folder`` won't disappear).
Please, refer to the Docker documentation for other use cases.

**WARNING**
If you are using *macos* you could encounter some disk space issues.
Please check `this <https://docs.docker.com/docker-for-mac/space/>`_ and `also this <https://djs55.github.io/jekyll/update/2017/11/27/docker-for-mac-disk-space.html>`_ on how to manage disk space.

Vagrant
+++++++

**NOTE**
Only required for users that want to use a *Singularity*
container on a *macos* and *Microsoft Windows* machine.

All users, regardless of their operative systems, can use protopipe and its interface via
`Vagrant <https://www.vagrantup.com/>`_. 

The *VagrantFile* provided with the interface code allows to download a virtual 
machine in form of a *Vagrant box* which will host the actual container.

The user needs only to edit a couple lines to link the source codes of the
pipeline and its interface.

The *VagrantFile* will create automatically also the ``shared_folder``
used by the interface to setup the analysis.

Singularity
+++++++++++

**WARNING**
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

The DIRAC grid certificate should be already available since *Singularity* mounts the user's home by default.
For more details, please check e.g. `system-defined bind paths <https://sylabs.io/guides/3.8/user-guide/bind_paths_and_mounts.html#system-defined-bind-paths>`_.

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

**WARNING**
The recipe ``CTADIRAC_singularity`` is maintained by the author; if any bug arises,
reverting to the methods described above (if possible) will provide you with a working environment.

If you have root privileges you can just build your own image with,

``singularity build ctadirac_client_latest.sif CTADIRAC_singularity``

otherwise you have to either,

- revert to the ``--fakeroot`` mode 
  (use it also to enter the container just like the methods above)

- build the image remotely at ``https://cloud.sylabs.io`` using the ``--remote`` flag
  (for this you will need to interface with that servce to generate an access token)
