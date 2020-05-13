.. _perf:

perf
====

Introduction
------------

The ``protopipe.perf`` module contains tools used to,

* handle the determination of the best-cutoffs to separate gamma and
  the background (protons + electrons),
* produce the instrument response functions (IRFs),
* estimate the sensitivity.

Details
-------

The following responses are computed:

 * Effective area as a function of true energy
 * Migration matrix as a function of true and reconstructed energy
 * Point spread function computed with a 68 % radius containment as a function of reconstructed energy
 * Background rate as a function of reconstructed energy

Responses
^^^^^^^^^

Effective area
""""""""""""""
The collection area, which is proportional to the gamma-ray efficiency
of detection, is computed as a function of the true energy. The events which
are considered are the one passing the threshold of the best cutoff plus
the angular cuts.

Energy migration matrix
"""""""""""""""""""""""
The migration matrix, ratio of the reconstructed energy over the true energy
as a function of the true energy, is computed with the events passing the
threshold of the best cutoff plus the angular cuts.
In order to be able to use the energy dispersion with Gammapy_
to compute the sensitvity we artificially created fake offset bins.
I guess that Gammapy_ should be able to reaf IRF with single offset.

Point spread function
"""""""""""""""""""""
Here we do not really need the PSF to compute the sensitivity, since the angular
cuts are already applied to the effective area, the energy migration matrix
and the background.
I chose to represent the PSF with a containment radius of 68 % as a function
of reconstructed energy as a simple HDU.
The events which are considered are the one passing the threshold of
the best cutoff.

Background rate
"""""""""""""""
The question to consider whether the bakground is an IRF or not. Since here it
is needed to estimate the sensitivity of the instrument we consider it is included
in the IRFs.
Here a simple HDU containing the background (protons + electrons) rate as a
function of the reconstructed energy is generated.
The events which are considered are the one passing the threshold of
the best cutoff and the angular cuts.

Sensitivity
^^^^^^^^^^^

The sensitivity is computed using Gammapy_.

Proposals for improvements and/or fixes
---------------------------------------

.. note::

  This section has to be moved to the repository as a set of issues.

.. note::

  Some of these points are relevant for the recently established
  `IRF working group <https://forge.in2p3.fr/projects/instrument-response-functions/wiki>`_
  of the CTA/ASWG consortium section.

* `Data format for IRFs <https://gamma-astro-data-formats.readthedocs.io/>`_
* Propagation and reading SIMTEL informations (meta-data, histograms)
  directly in the DL2
* Implement optimisation on the number of telescopes to consider an event
* We should generate the recommended IRF, e.g. parametrised as what? Apparently
  there are multiple solutions
  (see `here, <https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/psf/index.html>`_),
* implement the `Angular cut values, <https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/point_like/index.html>`_


Reference/API
-------------

.. automodapi:: protopipe.perf
    :no-inheritance-diagram:
    :include-all-objects:

.. _Gammapy: https://gammapy.org/
