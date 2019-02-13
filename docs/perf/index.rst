.. _perf:

====
perf
====

Introduction
============
The perf module contains classes that are used to estimate the performance of the
instrument. There are tools to handle the determination of the best-cutoffs
to separate gamma and the background (protons + electrons), to produce the
instrument response functions (IRFs) and to estimate the sensitivity.

The following responses are computed:
 * Effective area as a function of true energy
 * Migration matrix as a function of true and reconstructed energy
 * Point spread function computed with a 68 % radius containment as a function of reconstructed energy
 * Background rate as a function of reconstructed energy

The point-like source sensitivity is estimated with the gammapy_
library. We describe below how to estimate the performance of the instruments
and we describe in details how it is done.

How to?
=======
In order to estimate performance, you need lists of events at the
DL2 level, e.g. events with a minimal number of information:
 * Direction
 * True energy
 * Reconstructed energy
 * Score/gammaness

Three different tables of events, in HDF5_ format, are needed in order to estimate
the performance of the instruments:
 * Gamma-rays, considered as signal
 * Protons, considered as a source of diffuse background
 * Electrons, considered as a source of diffuse background

A single script called `make_performance.py` is used to estimate the performance:

.. code-block:: bash

    >$ ./make_performance.py --help
    usage: make_performance.py [-h] --config_file CONFIG_FILE [--wave | --tail]

An example configuration file is shown below with some comments:

.. code-block:: yaml

    general:
     # Directory with input data file
     indir: '/Users/julien/Documents/WorkingDir/Tools/python/protopipe/ana/prod_full_array_north_zen20_az0_complete/output/dl2/'
     # Template name for input file
     template_input_file: 'dl2_{}_{}_merged.h5'  # will be filled with mode and particle type
     # Directory for output files
     outdir: '/Users/julien/Documents/WorkingDir/Tools/python/protopipe/ana/prod_full_array_north_zen20_az0_complete/output/perf_mult3/'
     # Output table name
     output_table_name: 'table_best_cutoff'

    analysis:
     # Additional cut on data
     cut_on_data: 'NTels_reco >= 3'
     # Theta square cut optimisation (opti, fixed, r68)
     thsq_opt:
      type: 'r68'
      value: 0.2  # In degree, necessary for type fixed
     # Normalisation between ON and OFF regions
     alpha: 0.2
     # Observation time to estimate best cuts corresponding to best sensitivity
     obs_time:
      value: 50
      unit: 'h'
     min_sigma: 5  # minimal number of sigma
     min_excess: 10  # minimal number of excess events (nsig > min_excess)
     bkg_syst: 0.05  # percentage of bkg sytematics (nsig > bkg_syst * n_bkg)
     # Binning in reco energy (bkg rate, migration matrix)
     ereco_binning:  # TeV
      emin: 0.012589254
      emax: 199.52623
      nbin: 21
     # Binning for true energy (eff area, migration matrix, PSF)
     etrue_binning:  # TeV
      emin: 0.019952623
      emax: 199.52623
      nbin: 42

    # Information about simulation. In the future, everything should be store
    # in the input files (as meta data and as histogram)
    particle_information:
     # Simulated gamma-rays
     gamma:
      n_events_per_file: 1000000  #  number of files, 10**5 * 10
      e_min: 0.003  # energy min in TeV
      e_max: 330  # energy max in TeV
      gen_radius: 1400  # maximal impact parameter in meter
      diff_cone: 0  # diffuse cone, 0 or point-like, in degree
      gen_gamma: 2  # spectral index for input spectra
     # Simulated protons
     proton:
      n_events_per_file: 4000000  #  number of files, 2 * 10**5 * 20
      e_min: 0.004  # energy min in TeV
      e_max: 600  # energy max in TeV
      gen_radius: 1900  # maximal impact parameter in meter
      diff_cone: 10  # diffuse cone, 0 or point-like, in degree
      gen_gamma: 2  # spectral index for input spectra
      offset_cut: 1.  # maximum offset to consider particles
     # Simulated electrons
     electron:
      n_events_per_file: 2000000  #  number of files, 10**5 * 20
      e_min: 0.003  # energy min in TeV
      e_max: 330  # energy max in TeV
      gen_radius: 1900  # maximal impact parameter in meter
      diff_cone: 10  # diffuse cone, 0 or point-like, in degree
      gen_gamma: 2  # spectral index for input spectra
      offset_cut: 1.  # maximum offset to consider particles

    column_definition:
     # Column name for true energy
     mc_energy: 'mc_energy'
     # Column name for reconstructed energy
     reco_energy: 'reco_energy'
     # Column name for the angular distance in the camera between the true
     # position and the reconstructed position
     angular_distance_to_the_src: 'xi'
     # Column name for classification output
     classification_output:
      name: 'gammaness'
      range: [0, 1]  # needed to bin data and for diagnostic plots

Determination of the best cutoffs
==================================


Weighting of events
-------------------
salut ici :math:`\frac{ \sum_{t=0}^{N}f(t,k) }{N}` parce que

Best cutoffs
------------


Cutoffs application
-------------------


Diagnostics
-----------


How to produce IRFs
===================

Response of the instrument
--------------------------

Sensitivity
-----------

What could be improved?
=======================
 * `Data format for IRFs <https://gamma-astro-data-formats.readthedocs.io/>`_

Reference/API
=============

.. automodapi:: protopipe.perf
   :no-inheritance-diagram:

.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
.. _gammapy: https://gammapy.org/
.. _data format: https://gamma-astro-data-formats.readthedocs.io/
