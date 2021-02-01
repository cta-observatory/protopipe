.. _optimization_cuts_IRFs:

Optimized cuts and IRFs
=======================

In order to estimate performance, you need lists of events at the
DL2 level produced at the step :ref:`DL2`, e.g. events with a minimal number of information:

 * Direction
 * True energy
 * Reconstructed energy
 * Score/gammaness

Three different tables of events, in HDF5_ format, are needed in order to estimate
the performance of the instruments:

 * Gamma-rays, considered as signal
 * Protons, considered as a source of diffuse background
 * Electrons, considered as a source of diffuse background

The script ``protopipe.scripts.make_performance.py`` is used as follows:

.. code-block::

    usage: make_performance.py [-h] --config_file CONFIG_FILE --obs_time OBS_TIME
                               [--wave | --tail]

    Make performance files

    optional arguments:
      -h, --help            show this help message and exit
      --config_file CONFIG_FILE
      --obs_time OBS_TIME   Observation time, should be given as a string, value
                            and astropy unit separated by an empty space
      --wave                if set, use wavelet cleaning
      --tail                if set, use tail cleaning, otherwise wavelets

You can use ``protopipe/aux/scripts/multiple_performances.sh`` to produce
multiple IRFs for different observation times.

The configuration file for this step is ``performance.yaml``, here shown with some comments:

.. code-block:: yaml

    general:
     # Directory with input data file
     indir: ''
     # Template name for input file
     template_input_file: 'dl2_{}_{}_merged.h5'  # will be filled with mode and particle type
     # Directory for output files
     outdir: ''
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

.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
