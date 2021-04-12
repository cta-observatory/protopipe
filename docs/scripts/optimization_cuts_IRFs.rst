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
 
*protopipe* currently provides the DL2-to-DL3 step as performed by the *EventDisplay*
historical pipeline.
Additional scripts can be added as well.

The script ``protopipe.scripts.make_performance_EventDisplay.py`` is used as follows:

.. code-block::

  usage: protopipe-DL3-EventDisplay [-h] --config_file CONFIG_FILE
                                  [--wave | --tail]

  Make performance files

  optional arguments:
    -h, --help            show this help message and exit
    --config_file CONFIG_FILE
    --wave                if set, use wavelet cleaning
    --tail                if set, use tail cleaning (default)
    
The last two options can be ignored.

The configuration file for this step is ``performance.yaml``, here an example:

.. code-block:: yaml

  general:
   # Directory with input data file
   # [...] = your analysis local full path OUTSIDE the Vagrant box
   indir: '[...]/shared_folder/analyses/v0.4.0_dev1/data/DL2'
   # Template name for output file
   prod: 'Prod3b'
   site: 'North'
   array: 'baseline_full_array'
   zenith: '20deg'
   azimuth: '180deg' # 0deg -> north 180deg -> south
   template_input_file: 'DL2_{}_{}_merged.h5' # filled with mode and particle type
   # Directory for output files
   outdir: '[...]/shared_folder/analyses/v0.4.0_dev1/data/DL3'

  analysis:
   obs_time:
     value: 50
     unit: 'h'
   cut_on_multiplicity: 4
   # Normalisation between ON and OFF regions
   alpha: 0.2

   # Radius to use for calculating bg rate
   max_bg_radius: 1.

  particle_information:
   gamma:
    num_use: 10
    num_showers: 100000
    e_min: 0.003
    e_max: 330
    gen_radius: 1400
    gen_gamma: -2
    diff_cone: 0

   proton:
    num_use: 20
    num_showers: 200000
    e_min: 0.004
    e_max: 600
    gen_radius: 1900
    gen_gamma: -2
    diff_cone: 10

   electron:
    num_use: 20
    num_showers: 100000
    e_min: 0.003
    e_max: 330
    gen_radius: 1900
    gen_gamma: -2
    diff_cone: 10

.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/
